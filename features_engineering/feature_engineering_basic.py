import json
import numpy as np
import pandas as pd
from collections import Counter

# --- Configuration & Helper Functions ---

def weighted_frame_distance(frame1, frame2, weights):
    """
    Calculates the weighted distance between two single frames.
    weights: a numpy array of the same length as the frames.
    """
    # Using squared difference multiplied by weight
    # This allows you to say "flex8 is 5x more important than flex9"
    diff = (frame1 - frame2) ** 2
    weighted_diff = diff * weights
    return np.sqrt(np.sum(weighted_diff))

def manual_weighted_dtw(seq1, seq2, weights):
    """
    Full manual implementation of DTW with custom weights.
    seq1: Array of shape (N, num_features) - e.g., the sliding window
    seq2: Array of shape (M, num_features) - e.g., the Gold Standard
    weights: Array of shape (num_features,)
    """
    n = len(seq1)
    m = len(seq2)
    
    # 1. Initialize the Cost Matrix with infinity
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    
    # 2. Base case: distance at the starting point is 0
    dtw_matrix[0, 0] = 0
    
    # 3. Fill the matrix using dynamic programming
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Calculate cost between the current frames
            # Note: index i-1 and j-1 because the matrix is 1-indexed
            cost = weighted_frame_distance(seq1[i-1], seq2[j-1], weights)
            
            # The cumulative cost is the current cost plus the minimum 
            # of the three possible previous steps (match, insert, delete)
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # Insertion
                dtw_matrix[i, j-1],    # Deletion
                dtw_matrix[i-1, j-1]   # Match (Diagonal)
            )
            
    # The final DTW distance is the value at the top-right corner
    # Normalized by the length of the path (n + m) to keep values comparable
    return dtw_matrix[n, m] / (n + m)


def get_gold_segment(raw_frames, window_size, overlap_ratio=0.75):
    n_frames = len(raw_frames)

    # Case A: Sign is SHORTER than or equal to window_size
    if n_frames <= window_size:
        # We want to augment the data by placing the sign at the middle of the window.
        start = (window_size - n_frames) // 2
        window = [get_rest_frame()] * window_size
        for i in range(n_frames):
            window[start + i] = raw_frames[i]
        return np.array([flatten_frame(f) for f in window])

    # Case B: Sign is LONGER than window_size
    else:
        return np.array([flatten_frame(f) for f in raw_frames])


def get_segments(raw_frames, window_size, overlap_ratio=0.75):
    segments = []
    n_frames = len(raw_frames)
    
    # Calculate stride based on your desired overlap
    stride = max(1, int(window_size * (1 - overlap_ratio)))

    # Case A: Sign is SHORTER than or equal to window_size
    if n_frames <= window_size:
        # We want to augment the data by placing the sign at the 
        # beginning, middle, and end of the window.
        # This teaches the SVM 'Timing Invariance' without losing data.
        starts = [0, (window_size - n_frames) // 2, window_size - n_frames]
        
        for start in sorted(list(set(starts))):
            window = [get_rest_frame()] * window_size
            for i in range(n_frames):
                window[start + i] = raw_frames[i]
            segments.append(np.array([flatten_frame(f) for f in window]))

    # Case B: Sign is LONGER than window_size
    else:
        # Use a standard overlapping sliding window to chop the long sign.
        # This ensures every part of the long gesture is seen by the model.
        for start in range(0, n_frames - window_size + 1, stride):
            window = raw_frames[start : start + window_size]
            segments.append(np.array([flatten_frame(f) for f in window]))
            
        # Ensure the tail end of the sign is captured
        if (n_frames - window_size) % stride != 0:
            window = raw_frames[n_frames - window_size : n_frames]
            segments.append(np.array([flatten_frame(f) for f in window]))
            
    return segments

def get_rest_frame():
    """Returns a 'rest' frame based on specified parameters."""
    flex_keys = [f"flex{i}" for i in range(8, 16)]
    rest_flex = {k: {"curl": 0.0} for k in flex_keys}
    rest_pads = [{"n": f"PAD_CH{i}", "z": -1, "r": 0} for i in range(7)]
    return {"flex": rest_flex, "pads": rest_pads}

def flatten_frame(frame):
    """Converts the nested JSON frame into a flat numpy array for DTW/Features."""
    # Flex sensors 8-15
    flex_vals = [frame["flex"][f"flex{i}"]["curl"] for i in range(8, 16)]
    # Pad values (z and r)
    pad_vals = []
    for p in frame["pads"]:
        pad_vals.extend([p["z"], p["r"]])
    return np.array(flex_vals + pad_vals)

# --- 0. Load Data ---

file_path = 'data/interim/all_sessions.json'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# 1. Selection of Top 20 Labels
label_counts = Counter([item['label'] for item in raw_data])
top_20_labels = [label for label, count in label_counts.most_common(20)]
print(f"Top 20 labels: {top_20_labels}")

# --- 2. Window Size & Parameters ---

lengths = [len(item['frames']) for item in raw_data]
window_size = int(np.quantile(lengths, 0.9))
overlap = 0.75
stride = int(window_size * (1 - overlap))

print(f"Calculated Window Size (90th Quantile): {window_size} frames")
print(f"Stride for 75% overlap: {stride} frames")

#These to be changed based on experimentation and domain knowledge.
weights = np.ones(22)
#weights[0] = 5.0  # Give flex8 (index 0) high priority
#weights[8] = 2.0  # Give first pressure pad (index 8) medium priority

# --- 3. Feature Extraction & Processing ---

processed_features = []

# Extract Gold Standards for the Top 20
gold_standards_20 = {}
for item in raw_data:
    label = item['label']
    if label in top_20_labels and label not in gold_standards_20:
        gold_standards_20[label] = get_gold_segment(item['frames'], window_size, overlap_ratio=overlap)
    if len(gold_standards_20) == 20:
        break

# Second pass: Augmentation, Windowing, and Feature Extraction
for item in raw_data:
    label = item['label']
    raw_frames = item['frames']
    label_segments = get_segments(raw_frames, window_size, overlap_ratio=overlap)
    
    # Data Augmentation: Shifting and Padding
    for numeric_window in label_segments:

        
        # a) Flex Stats (Indices 0-7 in flattened frame)
        flex_data = numeric_window[:, :8]
        flex_means = np.mean(flex_data, axis=0)
        flex_vars = np.var(flex_data, axis=0)
        
        # b) Pad Stats (Indices 8-21: Z is at 8, 10, 12... R is at 9, 11, 13...)
        pad_data = numeric_window[:, 8:]
        max_z_vals = []
        corr_r_vals = []
        for i in range(0, 14, 2):
            z_col = pad_data[:, i]
            r_col = pad_data[:, i+1]
            max_idx = np.argmax(z_col)
            max_z_vals.append(z_col[max_idx])
            corr_r_vals.append(r_col[max_idx])
            
        # c) Total Energy of Changes (Flex only)
        # Sum of absolute differences between consecutive frames
        flex_energy = np.sum(np.abs(np.diff(flex_data, axis=0)), axis=0)
        

        # --- Construct Feature Row ---
        feature_row = {
            "label": label,
        }

        for anchor_label in top_20_labels:
            dist = manual_weighted_dtw(numeric_window, gold_standards_20[anchor_label], weights)
            feature_row[f"dtw_{anchor_label}"] = dist

        # Add flex features
        for i in range(8):
            feature_row[f"flex{i+8}_mean"] = flex_means[i]
            feature_row[f"flex{i+8}_var"] = flex_vars[i]
            feature_row[f"flex{i+8}_energy"] = flex_energy[i]
            
        # Add pad features
        for i in range(7):
            feature_row[f"pad{i}_max_z"] = max_z_vals[i]
            feature_row[f"pad{i}_corr_r"] = corr_r_vals[i]
            
        processed_features.append(feature_row)

# --- 4. Saving Results ---

df = pd.DataFrame(processed_features)
output_csv = "data/processed/extracted_features_basic.csv"
df.to_csv(output_csv, index=False)

print(f"Extraction complete. {len(processed_features)} augmented samples saved to {output_csv}.")


# Need to save the gold standards for inference
# We'll save them as a dictionary of lists (to be JSON serializable)
gs_serializable = {k: v.tolist() for k, v in gold_standards_20.items()}
with open('data/processed/gold_standards.json', 'w') as f:
    json.dump(gs_serializable, f)

print(f"Data saved. Total rows: {len(df)}. Total features: {len(df.columns)}")
print(df.head())