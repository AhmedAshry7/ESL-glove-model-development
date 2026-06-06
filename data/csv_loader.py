import os
import csv
import numpy as np
from collections import defaultdict

def load_sequences(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return []
        
    sequences = defaultdict(list)
    labels = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            sample_id = int(row[0])
            label = row[1]
            timestamp = float(row[3])
            features = [float(x) for x in row[4:]]
            
            labels[sample_id] = label
            sequences[sample_id].append([timestamp] + features)
            
    result = []
    for sid in sorted(sequences.keys()):
        result.append({
            "label": labels[sid],
            "frames": np.array(sequences[sid])
        })
    return result
