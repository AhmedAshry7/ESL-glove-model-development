
import json
import numpy as np
import pandas as pd
import joblib
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = 'models/best_svm_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ANCHORS_PATH = 'data/processed/dtw_anchors.json'
CLASSES_PATH = 'data/processed/classes.json'

WINDOW_SIZE = 40
STRIDE = 10
SMOOTHING_WINDOW = 4  # Queue size for temporal smoothing
COOLDOWN_N = 4        # Suppression windows after emission
CONFIDENCE_THRESHOLD = 0.6 # Minimum probability to accept classification

# --- HELPERS (Same as Training) ---

def flatten_frame(frame):
    flex_vals = [frame["flex"][f"flex{i}"]["curl"] for i in range(8, 16)]
    pad_vals = []
    for p in frame["pads"]:
        pad_vals.extend([p["z"], p["r"]])
    return np.array(flex_vals + pad_vals)

def weighted_frame_distance(frame1, frame2, weights):
    return np.sqrt(np.sum(((frame1 - frame2) ** 2) * weights))

def manual_weighted_dtw(seq1, seq2, weights):
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = weighted_frame_distance(seq1[i-1], seq2[j-1], weights)
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / (n + m)

def univariate_dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / (n + m)

def count_significant_changes(series, threshold=0.4):
    """
    Counts established trends and their reversals.
    - If a signal increases by >= threshold, it's 1 segment.
    - It stays in that segment as long as it keeps rising or doesn't drop by threshold.
    - If it drops from the peak by >= threshold, it counts as a 2nd segment (reversal).
    """
    if len(series) == 0: 
        return 0
        
    count = 0
    ref = series[0]
    direction = None  # Tracks current trend: 'up' or 'down'

    for val in series[1:]:
        if direction is None:
            # Establishing the first direction
            if val - ref >= threshold:
                count = 1
                direction = 'up'
                ref = val
            elif ref - val >= threshold:
                count = 1
                direction = 'down'
                ref = val
        elif direction == 'up':
            # In an upward trend, track the highest peak
            if val > ref:
                ref = val
            # If it drops from that peak by the threshold, it's a reversal
            elif ref - val >= threshold:
                count += 1
                direction = 'down'
                ref = val
        elif direction == 'down':
            # In a downward trend, track the lowest trough
            if val < ref:
                ref = val
            # If it rises from that trough by the threshold, it's a reversal
            elif val - ref >= threshold:
                count += 1
                direction = 'up'
                ref = val
                
    return count

def calculate_slope(series):
    """Calculates the linear regression slope of the sensor values over the window."""
    y = np.array(series)
    x = np.arange(len(y))
    # Standard linear regression slope formula
    if len(x) < 2: return 0
    slope = np.polyfit(x, y, 1)[0]
    return slope


def extract_features(numeric_window, anchors, gold_standards, weights):
    row = {}
    
    flex_data = numeric_window[:, :8]
    flex_means = np.mean(flex_data, axis=0)
    flex_vars = np.var(flex_data, axis=0)
    flex_max = np.max(flex_data, axis=0)
    flex_min = np.min(flex_data, axis=0)
    flex_sig_changes = [count_significant_changes(flex_data[:, i], threshold=0.4) for i in range(8)]
    flex_slopes = [calculate_slope(flex_data[:, i]) for i in range(8)]
    flex_energy = np.sum(np.abs(np.diff(flex_data, axis=0)), axis=0)
    
    # 1. DTW Features (Distance to each anchor)
    for anchor_label, anchor_seq in anchors.items():
        dist = manual_weighted_dtw(numeric_window, anchor_seq, weights)
        row[f'dtw_{anchor_label}'] = dist

    for i in range(8):
            f_idx = i + 8
            row[f"flex{f_idx}_mean"] = flex_means[i]
            row[f"flex{f_idx}_var"] = flex_vars[i]
            row[f"flex{f_idx}_energy"] = flex_energy[i]
            row[f"flex{f_idx}_max"] = flex_max[i]
            row[f"flex{f_idx}_min"] = flex_min[i]
            row[f"flex{f_idx}_sig_changes"] = flex_sig_changes[i]
            row[f"flex{f_idx}_slope"] = flex_slopes[i]
    
    # 2. Pad Features
    pad_data = numeric_window[:, 8:]
    valid_zs = [1, 3, 4, 5]
    pad_events = []
    for p_idx in range(7):
        z_col = pad_data[:, 2*p_idx]
        r_col = pad_data[:, 2*p_idx + 1]
        m_idx = np.argmax(z_col)
        row[f'pad{p_idx}_max_z'] = z_col[m_idx] + 1
        r_val = r_col[m_idx]
        row[f'pad{p_idx}_corr_r'] = 0 if r_val == 0 else (1 if r_val < 900 else 2)
        
        for vz in valid_zs:
            indices = np.where(z_col + 1 == vz)[0]
            first_occ = indices[0] if len(indices) > 0 else float('inf')
            pad_events.append(((p_idx, vz), first_occ))

    pad_events.sort(key=lambda x: x[1])
    event_orders = {}
    rank = 1
    for event, time in pad_events:
        if time != float('inf'):
            event_orders[event] = rank
            rank += 1
        else:
            event_orders[event] = 0
    
    for p_idx in range(7):
        for vz in valid_zs:
            row[f'pad{p_idx}_{vz}'] = event_orders[(p_idx, vz)]
            
    return row

# --- INFERENCE ENGINE ---

class RealTimeInference:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        with open(CLASSES_PATH, 'r') as f:
            self.classes = json.load(f)
        with open(ANCHORS_PATH, 'r') as f:
            data = json.load(f)
            self.anchors = {k: np.array(v) for k, v in data.items()}
        
        self.weights = np.ones(22)
        self.prob_queue = deque(maxlen=SMOOTHING_WINDOW)
        self.cooldown_counter = 0
        self.last_emitted = None

    def process_sequence(self, raw_frames):
        # Convert frames to numeric
        flat_frames = np.array([flatten_frame(f) for f in raw_frames])
        n_frames = len(flat_frames)
        emitted_words = []

        # Sliding window
        for start in range(0, n_frames - WINDOW_SIZE + 1, STRIDE):
            window = flat_frames[start : start + WINDOW_SIZE]
            
            # Feature extraction
            feat_dict = extract_features(window, self.anchors, self.weights)
            # Ensure column order matches training (alphabetical key order in dict to df usually works if trained that way)
            # Better: convert to df and reindex to match model features
            feat_df = pd.DataFrame([feat_dict])
            # Reorder columns to match the model's feature names
            feat_df = feat_df[self.model.feature_names_in_]
            
            X_scaled = self.scaler.transform(feat_df)
            
            # Stage 1: Thresholding
            probs = self.model.predict_proba(X_scaled)[0]
            max_prob = np.max(probs)
            
            if max_prob < CONFIDENCE_THRESHOLD:
                # Treat as background / nothing
                self.prob_queue.append(np.zeros_like(probs))
            else:
                self.prob_queue.append(probs)

            # Stage 2: Temporal Smoothing
            if len(self.prob_queue) == SMOOTHING_WINDOW:
                avg_probs = np.mean(self.prob_queue, axis=0)
                final_idx = np.argmax(avg_probs)
                final_prob = avg_probs[final_idx]
                
                if final_prob > CONFIDENCE_THRESHOLD:
                    predicted_word = self.classes[final_idx]

                    # Stage 3: Cooldown Suppression
                    if self.cooldown_counter > 0 and predicted_word == self.last_emitted:
                        self.cooldown_counter -= 1
                    else:
                        emitted_words.append(predicted_word)
                        self.last_emitted = predicted_word
                        self.cooldown_counter = COOLDOWN_N
                else:
                    if self.cooldown_counter > 0:
                        self.cooldown_counter -= 1
            
        return emitted_words

# --- RUN ---
if __name__ == "__main__":
    # Load inference data (100s of frames)
    # For demo, we'll just try to open a file named 'inference_data.json'
    try:
        with open('data/inference_data.json', 'r') as f:
            data = json.load(f)
            # If data is a list of frames:
            frames = data if isinstance(data, list) else data.get('frames', [])
            
            engine = RealTimeInference()
            output = engine.process_sequence(frames)
            print("Detected Signs:", " ".join(output))
    except FileNotFoundError:
        print("Please provide 'inference_data.json' with consecutive frames.")
