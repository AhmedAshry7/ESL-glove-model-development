import json
import numpy as np
import pandas as pd
from collections import Counter

# --- Configuration & Helper Functions ---

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

# --- 1. Load Data ---

file_path = 'data/interim/all_sessions.json'
with open(file_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

# --- 2. Window Size & Parameters ---

lengths = [len(item['frames']) for item in raw_data]
window_size = int(np.quantile(lengths, 0.9))
overlap = 0.75
stride = int(window_size * (1 - overlap))

print(f"Calculated Window Size (90th Quantile): {window_size} frames")
print(f"Stride for 75% overlap: {stride} frames")


# --- 3. Feature Extraction & Processing ---

processed_features = []
valid_zs = [1, 3, 4, 5]

# Second pass: Augmentation, Windowing, and Feature Extraction
for item in raw_data:
    label = item['label']
    raw_frames = item['frames']
    label_segments = get_segments(raw_frames, window_size, overlap_ratio=overlap)
    
    # Data Augmentation: Shifting and Padding
    for numeric_window in label_segments:

        
        # a) Flex Stats (Indices 0-7 in flattened frame)
        flex_data = numeric_window[:, :8]

        # b) Pad Stats (Indices 8-21: Z is at 8, 10, 12... R is at 9, 11, 13...)
        pad_data = numeric_window[:, 8:]
        
        max_z_vals, corr_r_vals = [], []
        for p_idx in range(7):
            z_col, r_col = pad_data[:, 2*p_idx], pad_data[:, 2*p_idx + 1]
            max_z_vals.append(z_col[np.argmax(z_col)])
            corr_r_vals.append(r_col[np.argmax(z_col)])
            
        
        # --- Construct Feature Row ---
        feature_row = {
            "label": label,
        }
        
        # Add flex features
        for i in range(8):
            f_idx = i + 8
            feature_row[f"flex{f_idx}"] = flex_data[:,i]
        # Add pad features (with the new Order column)
                
        
        for i in range(7):
                    r = corr_r_vals[i]
                    feature_row.update({f'pad{i}_max_z': max_z_vals[i] + 1, f'pad{i}_corr_r': 0 if r == 0 else (1 if r < 900 else 2)})
                    
        processed_features.append(feature_row)

# --- 4. Saving Results ---

df = pd.DataFrame(processed_features)
output_csv = "data/processed/formatted_data.csv"
df.to_csv(output_csv, index=False)

print(f"Extraction complete. {len(processed_features)} augmented samples saved to {output_csv}.")

print(f"Data saved. Total rows: {len(df)}. Total features: {len(df.columns)}")
print(df.head())