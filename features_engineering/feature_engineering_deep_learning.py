import argparse
import json
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

SEED = 42
FLEX_KEYS   = ["flex8", "flex9", "flex10", "flex11",
               "flex12", "flex13", "flex14", "flex15"]   # 8 sensors
N_PADS      = 7                                           # PAD_CH0 … PAD_CH6
N_FEATURES  = len(FLEX_KEYS) + N_PADS + N_PADS           # 22

TIME_WARP_VALUES   = [0.7, 0.85, 1.15, 1.3]   # stretch/compress factor range
SHIFT_VALUES = [0.5, 1, -0.5, -1]    # temporal shift as faction of W-frame window
LEAST_SHIFT=10 # minimum shift in frames to ensure a noticeable temporal shift
FLEX_NOISE_STD    = 0.01           # Gaussian noise σ on flex channels
FRAME_DROPOUT_MAX = 3              # max consecutive frames to zero out
AUGMENTATIONS_PER_SAMPLE = 8

def parse_recording(recording):
    """
    Convert one recording dict to a float32 array of shape (T, 22).
    """
    frames = recording["frames"]
    T = len(frames)
    arr = np.zeros((T, N_FEATURES), dtype=np.float32)

    for t, frame in enumerate(frames):
        flex = frame["flex"]
        for i, key in enumerate(FLEX_KEYS):
            arr[t, i] = float(flex[key]["curl"])

        pad_lookup = {p["n"]: p for p in frame["pads"]}
        for j in range(N_PADS):
            pad_name = f"PAD_CH{j}"
            pad = pad_lookup.get(pad_name, {"z": -1, "r": 0})
            arr[t, 8 + j]  = float(pad["z"])
            arr[t, 15 + j] = float(pad["r"])  

    return arr  


def load_all_recordings(json_path):
    """Load every JSON file and return (sequences, labels)."""
    sequences, labels = [], []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = [data]   
        for rec in data:
            arr = parse_recording(rec)
            sequences.append(arr)
            labels.append(rec["label"])
        print(f"  Loaded {len(data):3d} recordings from {json_path}")
    return sequences, labels


def normalize_pad_z(sequences):
    normalized = []
    for arr in sequences:
        out = arr.copy()
        out[:, 8:15] = (arr[:, 8:15] + 1) / 5 
        normalized.append(out)
    return normalized


def normalize_pad_r(sequences):
    normalized = []
    for arr in sequences:
        out = arr.copy()
        r_part = out[:, 15:22]
        binned_r = np.zeros_like(r_part)
        
        # Binning logic
        binned_r[(r_part > 0) & (r_part < 900)] = 1  
        binned_r[r_part >= 900] = 2                 
        
        out[:, 15:22] = binned_r / 2.0
        normalized.append(out)
    return normalized

def smooth_flex(sequences, window = 3):
    """
    Apply a rolling-mean smoothing of `window` frames to flex channels only. This suppresses sensor jitter without blurring pad events.
    """
    smoothed = []
    for arr in sequences:
        out = arr.copy()
        T = arr.shape[0]
        half = window // 2
        for t in range(T):
            lo = max(0, t - half)
            hi = min(T, t + half + 1)
            out[t, :8] = arr[lo:hi, :8].mean(axis=0)
        smoothed.append(out)
    return smoothed


def compute_window_size(sequences, percentile = 0.9):
    lengths = [arr.shape[0] for arr in sequences]
    window_size = int(np.quantile(lengths, percentile))
    overlap = 0.75
    return window_size


def pad_or_truncate(arr, W):
    """
    Shorter sequences: zero-padded at the END. Longer sequences: truncated from the end.
    """
    T = arr.shape[0]
    padded = np.zeros((W, N_FEATURES), dtype=np.float32)
    mask   = np.zeros(W, dtype=bool)

    if T >= W:
        padded[:] = arr[:W]
        mask[:] = True
    else:
        padded[:T] = arr
        mask[:T]   = True

    return padded, mask


# ─────────────────────────────────────────────
# Step 4: Data augmentation
# ─────────────────────────────────────────────

def augment_time_warp(arr, mask):
    """
    Stretch or compress the real (non-padded) part of the sequence by a
    random factor, then resample back to W frames via linear interpolation.
    """
    W = arr.shape[0]
    real_len = int(mask.sum())
    if real_len < 2:
        return arr.copy(), mask.copy()

    new_segments = []
    new_masks = []
    for factor in TIME_WARP_VALUES:
        
        new_real_len = max(2, min(W, int(round(real_len * factor))))

        # Interpolate the real portion
        real_part = arr[:real_len]                             # (real_len, 22)
        old_t = np.linspace(0, 1, real_len)
        new_t = np.linspace(0, 1, new_real_len)
        interp = interp1d(old_t, real_part, axis=0, kind="linear", fill_value="extrapolate")
        warped = interp(new_t).astype(np.float32)             # (new_real_len, 22)

        # Rebuild padded array and mask
        new_arr  = np.zeros((W, N_FEATURES), dtype=np.float32)
        new_mask = np.zeros(W, dtype=bool)
        actual   = min(new_real_len, W)
        new_arr[:actual]  = warped[:actual]
        new_mask[:actual] = True
        new_segments.append(new_arr)
        new_masks.append(new_mask)

    return new_segments, new_masks


def augment_flex_noise(arr, mask, rng):
    """
    Add small Gaussian noise to flex channels of real frames.
    """
    out = arr.copy()
    noise = rng.normal(0, FLEX_NOISE_STD, size=arr[:, :8].shape).astype(np.float32)
    noise[~mask] = 0.0               # zero noise on padding frames
    out[:, :8] = np.clip(arr[:, :8] + noise, 0.0, 1.0)
    return out, mask.copy()


def augment_temporal_shift(arr, mask):
    """
    Shift the sign forward and backward within the W-frame window.
    """
    W    = arr.shape[0]
    real_len = int(mask.sum())
    new_segments = []
    new_masks = []

    for shift_factor in SHIFT_VALUES:
        diff = max(LEAST_SHIFT, int(W-real_len))
        shift = int(shift_factor * diff)
        new_arr  = np.zeros((W, N_FEATURES), dtype=np.float32)
        new_mask = np.zeros(W, dtype=bool)

        if shift >= 0:
            # Shift right: sign starts at index `shift`
            end = min(W, shift + real_len)
            src_len = end - shift
            new_arr[shift:end]  = arr[:src_len]
            new_mask[shift:end] = mask[:src_len]
            new_segments.append(new_arr)
            new_masks.append(new_mask)
        else:
            # Shift left: discard |shift| frames from the start
            src_start = abs(shift)
            src_end   = min(arr.shape[0], src_start + W)
            dst_len   = src_end - src_start
            new_arr[:dst_len]  = arr[src_start:src_end]
            new_mask[:dst_len] = mask[src_start:src_end]
            new_segments.append(new_arr)
            new_masks.append(new_mask)
    
    return new_segments, new_masks


def augment_frame_dropout(arr, mask, rng):
    """
    Randomly zero out 1 to FRAME_DROPOUT_MAX consecutive real frames.
    This simulates momentary sensor glitches or transmission drops.
    """
    out      = arr.copy()
    real_idx = np.where(mask)[0]
    if len(real_idx) < 4:
        return out, mask.copy()

    n_drop = rng.integers(1, FRAME_DROPOUT_MAX + 1)
    # Pick a random start among real frames, leaving at least 1 real frame intact
    max_start = len(real_idx) - n_drop - 1
    if max_start < 1:
        return out, mask.copy()
    start_pos = rng.integers(0, max_start)
    drop_indices = real_idx[start_pos: start_pos + n_drop]
    out[drop_indices] = 0.0

    return out, mask.copy()  # mask unchanged — these frames ARE real, just corrupted


def generate_augmentations(arr, mask, n, rng):
    """
    Generate `n` augmented copies of one sample by randomly combining
    the four augmentation methods. 
    """
    all_final_arrays, all_final_masks = [], []

    # 'n' acts as the number of times we run the entire branching tree
    for _ in range(n):
        # Start with the original sample in a "processing list"
        current_batch = [(arr.copy(), mask.copy())]
        
        aug_fns = [
            augment_time_warp,
            augment_flex_noise,
            augment_temporal_shift,
            augment_frame_dropout
        ]
        
        # Shuffle the order of operations for variety
        order = rng.permutation(len(aug_fns))
        
        for idx in order:
            fn = aug_fns[idx]
            next_batch = []
            
            for a, m in current_batch:
                # Call the function (handling the rng argument mismatch)
                if fn in [augment_flex_noise, augment_frame_dropout]:
                    res_a, res_m = fn(a, m, rng)
                else:
                    res_a, res_m = fn(a, m)
                
                # Check if the function returned multiple variations (a list)
                if isinstance(res_a, list):
                    # Add all new variations to the next step
                    for sub_a, sub_m in zip(res_a, res_m):
                        next_batch.append((sub_a, sub_m))
                else:
                    # Function returned a single pair
                    next_batch.append((res_a, res_m))
            
            # The output of this augmentation becomes the input for the next one
            current_batch = next_batch
        
        # Unpack the final batch of this iteration into the main return lists
        for final_a, final_m in current_batch:
            all_final_arrays.append(final_a)
            all_final_masks.append(final_m)

    return all_final_arrays, all_final_masks

def encode_labels(labels, encoder_path):
    """
    Map string labels to integer class indices.
    """
    le = LabelEncoder()
    y  = le.fit_transform(labels).astype(np.int32)
    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)
    print(f"  Label encoder saved → {encoder_path}")
    print(f"  Classes ({len(le.classes_)}): {list(le.classes_)}")
    return y, le

def save_dataset(X_raw, y_raw, mask_raw, X_aug, y_aug, mask_aug, W, output_path, meta_output_path, metadata):
    """
    Save arrays to a compressed .npz file (NumPy's native multi-array format).

    Why .npz over alternatives?
    ─────────────────────────────
    • HDF5 (h5py): best for huge datasets with partial loading needs.
      Overkill here; adds a dependency.
    • Parquet / CSV: tabular formats, don't naturally hold 3-D tensors.
    • Pickle: fast but not portable across Python versions; no compression.
    • .npz: zero extra dependencies, ~50% compression via zlib, directly
      loadable by numpy/torch/tensorflow as np.load("dataset.npz").

    For datasets > ~1 GB you should migrate to HDF5 or TFRecord, but .npz
    is perfect for this project scale.
    """
    np.savez_compressed(
        output_path,
        X_raw   = X_raw,
        y_raw   = y_raw,
        mask_raw= mask_raw,
        X_aug   = X_aug,
        y_aug   = y_aug,
        mask_aug= mask_aug,
    )
    print(f"\n  Dataset saved → {output_path}.npz")
    print(f"    X_raw  : {X_raw.shape}   (original samples)")
    print(f"    X_aug  : {X_aug.shape}  (augmented samples)")

    meta_path = meta_output_path
    meta_full = {
        "W"            : W,
        "n_features"   : N_FEATURES,
        "flex_channels": list(range(0, 8)),
        "pad_z_channels": list(range(8, 15)),
        "pad_r_channels": list(range(15, 22)),
        "feature_names": (
            [f"flex_{k}_curl" for k in FLEX_KEYS] +
            [f"pad_ch{j}_z_norm" for j in range(N_PADS)] +
            [f"pad_ch{j}_r_norm" for j in range(N_PADS)]
        ),
        "n_classes"    : int(metadata["n_classes"]),
        "class_labels" : metadata["class_labels"],
        "n_raw"        : int(X_raw.shape[0]),
        "n_aug"        : int(X_aug.shape[0]),
        "aug_per_sample": metadata["aug_per_sample"],
        "augmentation_params": {
            "time_warp_range"  : TIME_WARP_VALUES,
            "shift_values"     : SHIFT_VALUES,
            "least_shift"      : LEAST_SHIFT,
            "flex_noise_std"   : FLEX_NOISE_STD,
            "frame_dropout_max": FRAME_DROPOUT_MAX,
        },
        "pad_z_normalization": {
            "method": "linear",
            "formula": "(z + 1) / 5  →  [0, 1]"
        },
        "pad_r_normalization": "Logical binning: 0=no contact, 1=thumb, 2=other finger; then scaled to [0, 1]",
        "flex_smoothing"     : "rolling mean window=3",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_full, f, ensure_ascii=False, indent=2)
    print(f"  Metadata saved   → {meta_path}")


rng = np.random.default_rng(SEED)

encoder_path = "models/label_encoder.pkl"
dataset_path = "data/processed/extracted_features_deep_learning"   # .npz appended by numpy
meta_output_path = "data/processed/extracted_features_deep_learning_metadata.json"
sequences, labels = load_all_recordings("data/interim/all_sessions.json")
print(f"  Total recordings loaded: {len(sequences)}")

sequences = smooth_flex(sequences, window=3) # Could be tuned or removed
sequences = normalize_pad_z(sequences)
sequences = normalize_pad_r(sequences)
# After this step every value in every sequence is in [0, 1]

W = compute_window_size(sequences, percentile=0.9)
print(f"  Window size W = {W} frames")

padded_seqs, masks = [], []
for arr in sequences:
    p, m = pad_or_truncate(arr, W)
    padded_seqs.append(p)
    masks.append(m)

X_raw    = np.stack(padded_seqs, axis=0)   # (N, W, 22)
mask_raw = np.stack(masks, axis=0)          # (N, W)
print(f"  X_raw shape: {X_raw.shape}")

y_raw, le = encode_labels(labels, encoder_path)

aug_arrays, aug_mask_list, aug_labels = [], [], []
aug_per_sample=AUGMENTATIONS_PER_SAMPLE

for i, (arr, m, label) in enumerate(zip(padded_seqs, masks, y_raw)):
    a_arrs, a_masks = generate_augmentations(arr, m, aug_per_sample, rng)
    aug_arrays.extend(a_arrs)
    aug_mask_list.extend(a_masks)
    aug_labels.extend([label] * aug_per_sample)

X_aug    = np.stack(aug_arrays, axis=0).astype(np.float32)
mask_aug = np.stack(aug_mask_list, axis=0)
y_aug    = np.array(aug_labels, dtype=np.int32)
print(f"  X_aug shape: {X_aug.shape}")

save_dataset(
    X_raw, y_raw, mask_raw,
    X_aug, y_aug, mask_aug,
    W, dataset_path, meta_output_path,
    metadata={
        "n_classes"     : len(le.classes_),
        "class_labels"  : list(le.classes_),
        "aug_per_sample": aug_per_sample,
    }
)

print("\n── Sanity checks ──────────────────────────────────────────────")
print(f"  X_raw  value range : [{X_raw.min():.4f},  {X_raw.max():.4f}]  (expect [0, 1])")
print(f"  X_aug  value range : [{X_aug.min():.4f},  {X_aug.max():.4f}]  (expect [0, 1])")
print(f"  y_raw unique labels: {np.unique(y_raw)}")
print(f"  Padding frames in X_raw: {(~mask_raw).sum()} / {mask_raw.size}")
print(f"  Per-class sample count (raw):")
for cls_idx, cls_name in enumerate(le.classes_):
    count = int((y_raw == cls_idx).sum())
    print(f"    [{cls_idx}] {cls_name}: {count} raw + {count * aug_per_sample} aug "
            f"= {count * (1 + aug_per_sample)} total")

print("\n✓ Preprocessing complete.\n")
print("  How to load in your training script:")
print("    data    = np.load('processed/dataset.npz')")
print("    X_train = np.concatenate([data['X_raw'], data['X_aug']], axis=0)")
print("    y_train = np.concatenate([data['y_raw'], data['y_aug']], axis=0)")
print("    masks   = np.concatenate([data['mask_raw'], data['mask_aug']], axis=0)")