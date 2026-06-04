import os
import csv
import numpy as np
from collections import defaultdict

def load_sequences_from_csv(csv_path: str) -> list:
    """Reads a CSV file and groups rows by sample_id.
    Returns a list of dicts: [{'label': str, 'frames': np.ndarray}, ...]
    where frames has shape (N, 57) [timestamp, f0...f55].
    """
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        return []
        
    sequences = defaultdict(list)
    labels = {}
    
    with open(csv_path, 'r') as f:
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
