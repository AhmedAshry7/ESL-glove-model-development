import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.stream_preprocessor import preprocess_stream, compute_normalization_stats
import config.pipeline_config as cfg
from data.csv_loader import load_sequences_from_csv

def _collect_all_raw_frames(train_csv_path: str) -> list:
    """
    Pass 1: collect preprocessed (but NOT yet normalized) frame arrays.
    Used to compute norm stats.
    """
    all_frames = []
    signs_data = load_sequences_from_csv(train_csv_path)
    
    for sign in signs_data:
        frames_data = sign.get("frames", [])
        if len(frames_data) == 0:
            continue
        _, v = preprocess_stream(
            frames_data[:, 0], frames_data[:, 1:],
            disabled_groups=cfg.DISABLED_FEATURE_GROUPS
        )
        if len(v) > 0:
            all_frames.append(v)

    return all_frames


def build_template_db(train_csv_path: str, out_dir: str, num_templates_per_sign=3):
    """
    Two-pass template builder:
    Pass 1  — compute per-channel z-score stats.
    Pass 2  — trim idle frames and save up to `num_templates_per_sign` templates per sign.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Clean stale templates from previous runs ────────────────────────────
    for f in os.listdir(out_dir):
        if f.endswith('.npz'):
            os.remove(os.path.join(out_dir, f))
    print("Cleaned stale templates from previous runs.")

    # ── Pass 1: compute & save normalization stats ──────────────────────────
    print("Pass 1: collecting frames to compute normalization statistics…")
    all_raw_frames = _collect_all_raw_frames(train_csv_path)
    if len(all_raw_frames) == 0:
        print("No frames collected. Please check the dataset path.")
        return
        
    norm_stats = compute_normalization_stats(all_raw_frames)
    np.savez_compressed(
        os.path.join(out_dir, "normalization_stats.npz"),
        mean=norm_stats['mean'],
        std=norm_stats['std']
    )
    print(f"  Saved normalization stats ({len(all_raw_frames)} sequences processed).")

    # ── Pass 2: build normalized templates ──────────────────────────────────
    print("Pass 2: building normalized templates…")
    extracted = {}  # {label: [array1, array2, ...]}
    
    signs_data = load_sequences_from_csv(train_csv_path)
    
    for sign in signs_data:
        label = sign.get("label")
        frames_data = sign.get("frames", [])
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
        
        extracted[label].append(v_trimmed)
        
    print("    Selecting medoid templates (most representative)...")
    from recognition.sdtw_engine import compute_static_dtw
    
    # Perform Medoid Selection and Dynamic Thresholding
    final_extracted = {}
    class_thresholds = {}
    for label, tmpls in extracted.items():
        if len(tmpls) <= num_templates_per_sign:
            final_extracted[label] = tmpls
            class_thresholds[label] = cfg.SDTW_DETECTION_THRESHOLD # fallback
        else:
            # Compute pairwise distance matrix
            n = len(tmpls)
            dist_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    d = compute_static_dtw(tmpls[i], tmpls[j], cfg.CHANNEL_WEIGHTS)
                    dist_mat[i, j] = d
                    dist_mat[j, i] = d
            
            # Find the 3 indices with the lowest sum of distances to all others
            sum_dists = np.sum(dist_mat, axis=1)
            best_indices = np.argsort(sum_dists)[:num_templates_per_sign]
            final_extracted[label] = [tmpls[i] for i in best_indices]
            
            # Compute dynamic threshold: max distance from any chosen template to any other sequence of the same class
            max_intra_dist = 0
            for best_idx in best_indices:
                for i in range(n):
                    if dist_mat[best_idx, i] > max_intra_dist:
                        max_intra_dist = dist_mat[best_idx, i]
            
            # Add a 20% margin to the maximum observed training distance, bounded between [0.2, 0.95]
            dyn_thresh = min(0.95, max(0.2, max_intra_dist * 1.20))
            class_thresholds[label] = dyn_thresh
            
    extracted = final_extracted
    
    # Save the dynamic thresholds
    import json
    with open(os.path.join(out_dir, "class_thresholds.json"), 'w') as f:
        json.dump(class_thresholds, f, indent=2)
    print("    Saved dynamic per-class thresholds.")
                    
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
    train_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed_csv", "train.csv")
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "templates")
    build_template_db(train_csv_path, out_dir)
