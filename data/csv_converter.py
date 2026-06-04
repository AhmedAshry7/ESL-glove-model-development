import os
import json
import csv
import random
from collections import defaultdict

def get_feature_names():
    features = []
    for side in ['right', 'left']:
        for finger in ['pinky', 'ring', 'middle', 'index']:
            features.extend([f"{side}_{finger}_{i}" for i in range(3)])
        features.extend([f"{side}_thumb_{i}" for i in range(4)])
        for imu in ['hand_imu', 'forearm_imu', 'upper_arm_imu']:
            features.extend([f"{side}_{imu}_w", f"{side}_{imu}_x", f"{side}_{imu}_y", f"{side}_{imu}_z"])
    return features

def convert_csv_split(raw_dir, processed_dir):
    
    all_rows = []
    samples = []
    sample_id = 0
    
    feature_names = get_feature_names()
    
    for filename in os.listdir(raw_dir):
        if filename.endswith('.json') and not filename.endswith('_meta.json'):
            filepath = os.path.join(raw_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                continue
                
            for sign in data:
                label = sign.get("label")
                frames = sign.get("frames", [])
                if len(frames) == 0:
                    continue
                
                samples.append((sample_id, label))
                for frame_idx, frame in enumerate(frames):
                    timestamp = frame[0]
                    features = frame[1:]
                    row = [sample_id, label, frame_idx, timestamp] + features
                    all_rows.append(row)
                    
                sample_id += 1
                
    if not all_rows:
        print("No data found to convert.")
        return
        
    print(f"Total frames extracted: {len(all_rows)}")
    print(f"Total unique signs: {len(samples)}")
    os.makedirs(processed_dir, exist_ok=True)

    # Mapping sample_ids by label
    label_to_samples = defaultdict(list)
    for sid, label in samples:
        label_to_samples[label].append(sid)
        
    train_sids = set()
    val_sids = set()
    test_sids = set()
    
    random.seed(42)
    for label, sids in label_to_samples.items():
        random.shuffle(sids)
        total = len(sids)
        
        train_end = int(total * 0.6)
        val_end = train_end + int(total * 0.20)
        
        train_sids.update(sids[:train_end])
        val_sids.update(sids[train_end:val_end])
        test_sids.update(sids[val_end:])
        
    print(f"Training sequences: {len(train_sids)}")
    print(f"Validation sequences: {len(val_sids)}")
    print(f"Testing sequences: {len(test_sids)}")
    
    train_rows = [row for row in all_rows if row[0] in train_sids]
    val_rows = [row for row in all_rows if row[0] in val_sids]
    test_rows = [row for row in all_rows if row[0] in test_sids]
    
    train_file = os.path.join(processed_dir, "train.csv")
    val_file = os.path.join(processed_dir, "val.csv")
    test_file = os.path.join(processed_dir, "test.csv")
    
    columns = ['sample_id', 'label', 'frame_index', 'timestamp'] + feature_names
    
    print("Saving CSV files...")
    for file_path, dataset_rows in [(train_file, train_rows), (val_file, val_rows), (test_file, test_rows)]:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(dataset_rows)
    
    print(f"Success! Data saved:")
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed_csv")
    convert_csv_split(raw_dir, processed_dir)

