import numpy as np
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pipeline_config import ROLLING_WINDOW_SIZE, N_FEATURES_PER_FRAME, PREFILTER_MIN_FILL_RATIO
import config.pipeline_config as cfg
from preprocessing.stream_preprocessor import preprocess_stream
from recognition.rolling_stats import RollingStatsBuffer

def load_norm_stats(model_dir: str) -> dict:
    """
    Load normalization statistics saved by build_template_db.
    Returns a dict with 'mean' and 'std' arrays, or None if not found.
    """
    stats_path = os.path.join(model_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        return None
    data = np.load(stats_path)
    return {'mean': data['mean'], 'std': data['std']}


def extract_prefilter_dataset(raw_dir: str, disabled_groups: list = [], norm_stats: dict = None):
    """
    Process all sessions in raw_dir. For each annotated sign, extract
    rolling stats (Option D) from the frames during the sign to use as
    training data for the Random Forest pre-filter.

    Args:
        raw_dir:        Directory containing training JSON/NPZ files.
        disabled_groups: Feature groups to zero out.
        norm_stats:     Normalization stats dict (from load_norm_stats).
                        If None, features are computed on raw-scale data
                        (not recommended — train and build_template_db
                        should always be run together).

    Returns: X (features, shape N×282), Y (labels, shape N,)
    """
    X = []
    Y = []

    if not os.path.exists(raw_dir):
        return np.array(X), np.array(Y)

    for filename in os.listdir(raw_dir):
        if filename.endswith('.json') and not filename.endswith('_meta.json'):
            file_path = os.path.join(raw_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    signs_data = json.load(f)
            except json.JSONDecodeError:
                continue

            if not isinstance(signs_data, list):
                continue

            for sign in signs_data:
                label = sign.get("label")
                frames_data = np.array(sign.get("frames", []))
                if len(frames_data) == 0:
                    continue

                timestamps = frames_data[:, 0]
                raw_frames = frames_data[:, 1:]

                # Preprocess + normalize this disjoint sign
                t, v = preprocess_stream(timestamps, raw_frames, disabled_groups, norm_stats=norm_stats)
                if len(v) == 0:
                    continue

                end_idx = len(v) - 1
                sample_points = np.linspace(min(5, end_idx), end_idx, 5, dtype=int)

                for pt in sample_points:
                    buffer = RollingStatsBuffer(window_size=ROLLING_WINDOW_SIZE, n_features=N_FEATURES_PER_FRAME)
                    window_start = max(0, pt - ROLLING_WINDOW_SIZE + 1)
                    for i in range(window_start, pt + 1):
                        buffer.push(v[i])

                    # Gate: only use sample points where the buffer is sufficiently full
                    if buffer.fill_ratio < PREFILTER_MIN_FILL_RATIO:
                        continue

                    features = buffer.get_stats_features()
                    if features is not None:
                        X.append(features)
                        Y.append(label)

        elif filename.endswith('.npz'):
            session_id = filename.replace('.npz', '')
            npz_path = os.path.join(raw_dir, filename)
            meta_path = os.path.join(raw_dir, f"{session_id}_meta.json")

            if not os.path.exists(meta_path):
                continue

            data = np.load(npz_path)
            raw_frames = data['frames']
            timestamps = data['timestamps']

            with open(meta_path, 'r') as f:
                meta = json.load(f)

            # Preprocess + normalize the entire session
            t, v = preprocess_stream(timestamps, raw_frames, disabled_groups, norm_stats=norm_stats)

            # Extract features around each annotation
            for ann in meta.get("annotations", []):
                label = ann["label"]
                start_raw_idx = ann["start"]
                end_raw_idx = ann["end"]

                start_time = (timestamps[start_raw_idx] - timestamps[0]) / 1e6
                end_raw_idx_safe = min(end_raw_idx - 1, len(timestamps) - 1)
                end_time = (timestamps[end_raw_idx_safe] - timestamps[0]) / 1e6

                start_idx = np.searchsorted(t, start_time)
                end_idx = np.searchsorted(t, end_time)
                end_idx = min(end_idx, len(v) - 1)

                if end_idx <= start_idx:
                    continue

                sample_points = np.linspace(start_idx + min(5, end_idx - start_idx), end_idx, 5, dtype=int)

                for pt in sample_points:
                    buffer = RollingStatsBuffer(window_size=ROLLING_WINDOW_SIZE, n_features=N_FEATURES_PER_FRAME)
                    window_start = max(0, pt - ROLLING_WINDOW_SIZE + 1)
                    for i in range(window_start, pt + 1):
                        buffer.push(v[i])

                    if buffer.fill_ratio < PREFILTER_MIN_FILL_RATIO:
                        continue

                    features = buffer.get_stats_features()
                    if features is not None:
                        X.append(features)
                        Y.append(label)

    return np.array(X), np.array(Y)


if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    norm_stats = load_norm_stats(model_dir)
    if norm_stats is None:
        print("Warning: normalization stats not found. Run build_template_db.py first.")
    X, Y = extract_prefilter_dataset(raw_dir, disabled_groups=cfg.DISABLED_FEATURE_GROUPS, norm_stats=norm_stats)
    print(f"Extracted {len(X)} training samples for pre-filter.")
    if len(X) > 0:
        print(f"Feature vector shape: {X.shape}")

