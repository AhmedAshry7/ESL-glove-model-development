import numpy as np
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pipeline_config import ROLLING_WINDOW_SIZE, N_FEATURES_PER_FRAME, PREFILTER_MIN_FILL_RATIO
import config.pipeline_config as cfg
from preprocessing.stream_preprocessor import preprocess_stream
from recognition.rolling_stats import RollingStatsBuffer
from data.csv_loader import load_sequences

def load_norm_stats(model_dir: str) -> dict:

    stats_path = os.path.join(model_dir, "templates", "normalization_stats.npz")
    if not os.path.exists(stats_path):
            return None
    data = np.load(stats_path)
    return {'mean': data['mean'], 'std': data['std']}


from scipy.interpolate import interp1d

def _interpolate_sequence(seq, target_len=50):
    T, F = seq.shape
    if T == target_len:
        return seq
    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, target_len)
    interpolator = interp1d(old_indices, seq, axis=0, kind='linear', fill_value="extrapolate")
    return interpolator(new_indices)

def extract_prefilter_dataset(train_csv_path: str, disabled_groups: list = [], norm_stats: dict = None):

    X = []
    Y = []

    signs_data = load_sequences(train_csv_path)

    for sign in signs_data:
        label = sign.get("label")
        frames_data = sign.get("frames", [])
        if len(frames_data) == 0:
            continue

        timestamps = frames_data[:, 0]
        raw_frames = frames_data[:, 1:]

        t, v = preprocess_stream(timestamps, raw_frames, disabled_groups, norm_stats=norm_stats)
        if len(v) < 5:
            continue
            
        # Instead of pushing arbitrarily and sub-sampling, interpolate to exactly ROLLING_WINDOW_SIZE
        normalized_v = _interpolate_sequence(v, target_len=ROLLING_WINDOW_SIZE)

        buffer = RollingStatsBuffer(window_size=ROLLING_WINDOW_SIZE, n_features=N_FEATURES_PER_FRAME)
        
        # Push the perfectly normalized sequence into the buffer
        for frame in normalized_v:
            buffer.push(frame)

        features = buffer.calc_features()
        if features is not None:
            X.append(features)
            Y.append(label)

    return np.array(X), np.array(Y)


if __name__ == "__main__":
    train_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed_csv", "train.csv")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    norm_stats = load_norm_stats(model_dir)
    if norm_stats is None:
        print("Warning: normalization stats not found. Run build_template_db.py first.")
    X, Y = extract_prefilter_dataset(train_csv_path, disabled_groups=cfg.DISABLED_FEATURE_GROUPS, norm_stats=norm_stats)
    print(f"Extracted {len(X)} training samples for pre-filter.")
    if len(X) > 0:
        print(f"Feature vector shape: {X.shape}")

