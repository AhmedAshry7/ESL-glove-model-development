import numpy as np
import os
import json
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.stream_preprocessor import preprocess_stream, compute_normalization_stats
import config.pipeline_config as cfg

def _collect_all_raw_frames(raw_dir: str) -> list:
    """
    Pass 1: iterate all training JSON/NPZ files and collect preprocessed
    (but NOT yet normalized) frame arrays.  Used to compute norm stats.
    """
    all_frames = []
    if not os.path.exists(raw_dir):
        return all_frames

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
                frames_data = np.array(sign.get("frames", []))
                if len(frames_data) == 0:
                    continue
                _, v = preprocess_stream(
                    frames_data[:, 0], frames_data[:, 1:],
                    disabled_groups=cfg.DISABLED_FEATURE_GROUPS
                )
                if len(v) > 0:
                    all_frames.append(v)

        elif filename.endswith('.npz'):
            session_id = filename.replace('.npz', '')
            meta_path = os.path.join(raw_dir, f"{session_id}_meta.json")
            if not os.path.exists(meta_path):
                continue
            data = np.load(os.path.join(raw_dir, filename))
            _, v = preprocess_stream(
                data['timestamps'], data['frames'],
                disabled_groups=cfg.DISABLED_FEATURE_GROUPS
            )
            if len(v) > 0:
                all_frames.append(v)

    return all_frames


def build_template_db(raw_dir: str, out_dir: str, num_templates_per_sign=3):
    """
    Two-pass template builder:

    Pass 1  — collect all preprocessed (raw, un-normalized) frames from
              every training recording and compute per-channel z-score stats.
              Stats are saved to ``out_dir/normalization_stats.npz``.

    Pass 2  — re-read each recording, apply normalization, trim idle frames,
              and save up to ``num_templates_per_sign`` templates per sign.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Pass 1: compute & save normalization stats ──────────────────────────
    print("Pass 1: collecting frames to compute normalization statistics…")
    all_raw_frames = _collect_all_raw_frames(raw_dir)
    norm_stats = compute_normalization_stats(all_raw_frames)
    np.savez_compressed(
        os.path.join(out_dir, "normalization_stats.npz"),
        mean=norm_stats['mean'],
        std=norm_stats['std']
    )
    print(f"  Saved normalization stats ({len(all_raw_frames)} recordings processed).")

    # ── Pass 2: build normalized templates ──────────────────────────────────
    print("Pass 2: building normalized templates…")
    extracted = {}  # {label: [array1, array2, ...]}
    
    if not os.path.exists(raw_dir):
        print(f"Directory {raw_dir} does not exist.")
        return
        
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
                
                t, v = preprocess_stream(timestamps, raw_frames, disabled_groups=cfg.DISABLED_FEATURE_GROUPS, norm_stats=norm_stats)
                if len(v) == 0:
                    continue
                
                # Trim IDLE frames to match ContinuousRecognizer behavior
                from recognition.activity_monitor import ActivityMonitor
                
                # Find active channels based on weights
                active_channels = np.where(cfg.CHANNEL_WEIGHTS > 0)[0]
                finger_indices = list(range(0, 16)) + list(range(28, 44))
                imu_indices = list(range(16, 28)) + list(range(44, 56))
                
                active_finger_channels = [i for i in active_channels if i in finger_indices]
                active_imu_channels = [i for i in active_channels if i in imu_indices]

                monitor = ActivityMonitor(
                    finger_thresh=cfg.FINGER_IDLE_THRESHOLD_NORM,
                    arm_thresh=cfg.ARM_IDLE_THRESHOLD,
                    idle_frames_req=cfg.IDLE_FRAMES_REQUIRED,
                    active_finger_channels=active_finger_channels,
                    active_imu_channels=active_imu_channels
                )
                active_frames = []
                for frame in v:
                    if monitor.feed(frame):
                        active_frames.append(frame)
                
                if len(active_frames) == 0:
                    continue
                    
                v_trimmed = np.array(active_frames)
                
                if label not in extracted:
                    extracted[label] = []
                
                if len(extracted[label]) < num_templates_per_sign:
                    extracted[label].append(v_trimmed)
                    
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
                
            t, v = preprocess_stream(timestamps, raw_frames, disabled_groups=cfg.DISABLED_FEATURE_GROUPS, norm_stats=norm_stats)
            
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
                
                if end_idx > start_idx:
                    if label not in extracted:
                        extracted[label] = []
                    
                    if len(extracted[label]) < num_templates_per_sign:
                        extracted[label].append(v[start_idx:end_idx])
                    
    for label, tmpls in extracted.items():
        save_dict = {f"tmpl_{i}": tmpl for i, tmpl in enumerate(tmpls)}
        np.savez_compressed(os.path.join(out_dir, f"{label}.npz"), **save_dict)
        print(f"  Saved {len(tmpls)} templates for '{label}'")

    # ── Pass 3: Compute Pairwise Channel Discrimination ─────────────────────
    print("Pass 3: Computing pairwise discriminative weights…")
    labels = list(extracted.keys())
    mean_poses = {}
    for label in labels:
        if len(extracted[label]) > 0:
            all_frames = np.vstack(extracted[label])
            mean_poses[label] = np.mean(all_frames, axis=0)
            
    discriminative_weights = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            lA, lB = labels[i], labels[j]
            if lA in mean_poses and lB in mean_poses:
                diff = np.abs(mean_poses[lA] - mean_poses[lB])
                sum_diff = np.sum(diff)
                if sum_diff > 0:
                    weights = diff / sum_diff
                else:
                    weights = np.ones(56) / 56.0
                
                # Multiply by global weights to keep IMU/finger priority intact
                w_final = weights * cfg.CHANNEL_WEIGHTS
                sum_final = np.sum(w_final)
                if sum_final > 0:
                    w_final /= sum_final
                    
                discriminative_weights[f"{lA}_vs_{lB}"] = w_final
                discriminative_weights[f"{lB}_vs_{lA}"] = w_final
                
    if discriminative_weights:
        np.savez_compressed(
            os.path.join(out_dir, "discriminative_weights.npz"),
            **discriminative_weights
        )
        print(f"  Saved discriminative weights for {len(discriminative_weights)//2} pairs.")

    print(f"\nDone. {len(extracted)} sign classes with normalized templates.")

if __name__ == "__main__":
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "templates")
    build_template_db(raw_dir, out_dir)
