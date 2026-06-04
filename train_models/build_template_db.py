import numpy as np
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.stream_preprocessor import preprocess_stream, compute_normalization_stats, normalize_frames
import config.pipeline_config as cfg
from recognition.activity_monitor import ActivityMonitor
from recognition.sdtw_engine import compute_static_dtw
from data.csv_loader import load_sequences

def collect_raw_frames(train_path: str) -> list:
    
    all_frames = []
    labels=[]
    signs_data = load_sequences(train_path)
    
    for sign in signs_data:
        frames_data = sign.get("frames", [])
        if len(frames_data) == 0:
            continue
        _, v = preprocess_stream(frames_data[:, 0], frames_data[:, 1:], disabled_groups=cfg.DISABLED_FEATURE_GROUPS)
        if len(v) > 0:
            all_frames.append(v)
            labels.append(sign.get("label"))
    return all_frames, labels


def build_template_db(train_path: str, template_dir: str, num_templates=3):

    os.makedirs(template_dir, exist_ok=True)

    for f in os.listdir(template_dir):
        if f.endswith('.npz'):
            os.remove(os.path.join(template_dir, f))
    print("Cleaned old templates from previous runs.")

    all_raw_frames, labels = collect_raw_frames(train_path)
    if len(all_raw_frames) == 0:
        print("No frames collected. Please check the dataset path.")
        return
        
    norm_stats = compute_normalization_stats(all_raw_frames)
    np.savez_compressed(
        os.path.join(template_dir, "normalization_stats.npz"),
        mean=norm_stats['mean'],
        std=norm_stats['std']
    )
    print(f"Saved normalization stats ({len(all_raw_frames)} sequences processed).")

    print("building normalized templates")
    extracted = {}
    
    for frames, label in zip(all_raw_frames, labels):
        v = normalize_frames(frames, norm_stats)

        
        # Trim IDLE frames to match inference behavior
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
        
    #Selecting most representative templates (medoids)    
    print("Selecting medoid templates")
    
    final_extracted = {}
    class_thresholds = {}
    for label, templates in extracted.items():
        if len(templates) <= num_templates:
            final_extracted[label] = templates
            class_thresholds[label] = cfg.SDTW_DETECTION_THRESHOLD
        else:
            n = len(templates)
            dist_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    d = compute_static_dtw(templates[i], templates[j], cfg.CHANNEL_WEIGHTS)
                    dist_mat[i, j] = d
                    dist_mat[j, i] = d
            
            sum_dists = np.sum(dist_mat, axis=1)
            best_indices = np.argsort(sum_dists)[:num_templates]
            final_extracted[label] = [templates[i] for i in best_indices]
            
            # Compute dynamic threshold: max distance from any chosen template to any other sequence of the same class
            longest_dist = 0
            for best_index in best_indices:
                for i in range(n):
                    if dist_mat[best_index, i] > longest_dist:
                        longest_dist = dist_mat[best_index, i]
            
            # Add a minimum 20% margin to the maximum observed training distance, bounded between [0.2, 0.75]
            class_thresholds[label] = min(0.75, max(0.2, longest_dist * 1.20))
            
    extracted = final_extracted
    
    with open(os.path.join(template_dir, "class_thresholds.json"), 'w') as f:
        json.dump(class_thresholds, f, indent=2)
    print("Saved dynamic per-class thresholds.")
                    
    for label, templates in extracted.items():
        save_dict = {f"template_{i}": template for i, template in enumerate(templates)}
        np.savez_compressed(os.path.join(template_dir, f"{label}.npz"), **save_dict)

    print("Computing pairwise discriminative weights…")
    labels = list(extracted.keys())
    mean_poses = {}
    for label in labels:
        if len(extracted[label]) > 0:
            all_frames = np.vstack(extracted[label])
            mean_poses[label] = np.mean(all_frames, axis=0)
            
    discriminative_weights = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            l1, l2 = labels[i], labels[j]
            if l1 in mean_poses and l2 in mean_poses:
                diff = np.abs(mean_poses[l1] - mean_poses[l2])
                sum_diff = np.sum(diff)
                if sum_diff > 0:
                    weights = diff / sum_diff
                else:
                    weights = np.ones(56) / 56.0
                
                w_final = weights * cfg.CHANNEL_WEIGHTS
                sum_final = np.sum(w_final)
                if sum_final > 0:
                    w_final /= sum_final
                    
                discriminative_weights[f"{l1}_vs_{l2}"] = w_final
                discriminative_weights[f"{l2}_vs_{l1}"] = w_final
                
    if discriminative_weights:
        np.savez_compressed(
            os.path.join(template_dir, "discriminative_weights.npz"),
            **discriminative_weights
        )
        print(f"Saved discriminative weights for {len(discriminative_weights)//2} pairs.")

    print(f"\nDone. {len(extracted)} sign classes with normalized templates.")

if __name__ == "__main__":
    train_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed_csv", "train.csv")
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "templates")
    build_template_db(train_path, template_dir)
