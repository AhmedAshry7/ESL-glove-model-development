import json
import numpy as np
import pandas as pd
import joblib
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = 'models/best_knn_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

WINDOW_SIZE = 40
STRIDE = 10
SMOOTHING_WINDOW = 4 
COOLDOWN_N = 4       
CONFIDENCE_THRESHOLD = 0.6 

# --- DTW METRICS (Must match training exactly) ---

def univariate_dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m] / (n + m)

def custom_dtw_combined_dist(x1, x2):
    """
    Custom distance metric used during training:
    - First 320 elements: 8 flex sensors (40 samples each)
    - Last 14 elements: pad max_z and corr_r values
    """
    total_dist = 0
    # DTW for flex sensors
    for i in range(8):
        s1 = x1[i*40 : (i+1)*40]
        s2 = x2[i*40 : (i+1)*40]
        total_dist += univariate_dtw_distance(s1, s2)
    
    # Euclidean for pad features
    pad1 = x1[320:]
    pad2 = x2[320:]
    total_dist += np.linalg.norm(pad1 - pad2)
    return total_dist

# --- HELPERS ---

def flatten_frame(frame):
    # Extracts raw numeric values from a single JSON frame
    flex_vals = [frame["flex"][f"flex{i}"]["curl"] for i in range(8, 16)]
    pad_vals = []
    for p in frame["pads"]:
        pad_vals.extend([p["z"], p["r"]])
    return np.array(flex_vals + pad_vals)

def extract_combined_vector(numeric_window):
    """
    Converts a 40-frame window into a single 334-element feature vector.
    Matches the training preprocessing logic.
    """
    # 1. Flex Sequences (320 elements)
    flex_data = numeric_window[:, :8] # shape (40, 8)
    # Transpose and flatten so it's [flex8_t0...flex8_t39, flex9_t0...]
    flex_vector = flex_data.T.flatten() 

    # 2. Pad Features (14 elements)
    pad_data = numeric_window[:, 8:] # shape (40, 14)
    pad_features = []
    for p_idx in range(7):
        z_col = pad_data[:, 2*p_idx]
        r_col = pad_data[:, 2*p_idx + 1]
        m_idx = np.argmax(z_col)
        
        max_z = z_col[m_idx] + 1
        r_val = r_col[m_idx]
        corr_r = 0 if r_val == 0 else (1 if r_val < 900 else 2)
        
        pad_features.extend([max_z, corr_r])
            
    return np.concatenate([flex_vector, np.array(pad_features)])

# --- INFERENCE ENGINE ---

class RealTimeInference:
    def __init__(self):
        # We must ensure custom_dtw_combined_dist is in the global scope 
        # for joblib to load the model correctly.
        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.classes = self.model.classes_ # Get labels from the trained model
        
        self.prob_queue = deque(maxlen=SMOOTHING_WINDOW)
        self.cooldown_counter = 0
        self.last_emitted = None

    def process_sequence(self, raw_frames):
        # Convert list of JSON frames to a numeric matrix
        flat_frames = np.array([flatten_frame(f) for f in raw_frames])
        n_frames = len(flat_frames)
        emitted_words = []

        # Sliding window approach
        for start in range(0, n_frames - WINDOW_SIZE + 1, STRIDE):
            window = flat_frames[start : start + WINDOW_SIZE]
            
            # Create the 334-feature vector (Flex sequences + Pad stats)
            feat_vector = extract_combined_vector(window)
            
            # Scale the vector (expects 2D input)
            X_scaled = self.scaler.transform(feat_vector.reshape(1, -1))
            
            # Stage 1: Thresholding
            probs = self.model.predict_proba(X_scaled)[0]
            max_prob = np.max(probs)
            
            if max_prob < CONFIDENCE_THRESHOLD:
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

                    # Stage 3: Cooldown Suppression (Debouncing)
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
    try:
        with open('data/inference_data.json', 'r') as f:
            data = json.load(f)
            frames = data if isinstance(data, list) else data.get('frames', [])
            
            engine = RealTimeInference()
            output = engine.process_sequence(frames)
            print("\nDetected Signs:", " ".join(output))
    except FileNotFoundError:
        print("Error: 'data/inference_data.json' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")