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
        if len(v) == 0:
            continue

        end_index = len(v) - 1
        sample_points = np.linspace(min(5, end_index), end_index, 5, dtype=int)
        buffer = RollingStatsBuffer(window_size=ROLLING_WINDOW_SIZE, n_features=N_FEATURES_PER_FRAME)
        for point in sample_points:
            window_start = max(0, point - ROLLING_WINDOW_SIZE + 1)
            for i in range(window_start, point + 1):
                buffer.push(v[i])

            # Checking if buffer passes the min fill
            if buffer.fill_ratio < PREFILTER_MIN_FILL_RATIO:
                continue

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

