import sys
import os
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.stream_preprocessor import preprocess_stream
from inference.continuous_recognizer import ContinuousRecognizer

def test_pipeline_on_synthetic_data(data_file: str, meta_file: str, model_dir: str):
    print(f"Testing pipeline on {data_file}")
    
    # 1. Load Data
    data = np.load(data_file)
    raw_frames = data['frames']
    timestamps = data['timestamps']
    
    with open(meta_file, 'r') as f:
        meta = json.load(f)
        
    print(f"Raw frames: {len(raw_frames)}")
    print("Ground Truth Annotations:")
    for ann in meta.get("annotations", []):
        print(f"  - {ann['label']} [Frames {ann['start']}-{ann['end']}]")
        
    # 2. Preprocess
    t, v = preprocess_stream(timestamps, raw_frames, disabled_groups=[])
    print(f"Preprocessed frames (uniform 50Hz grid): {len(v)}")
    
    # 3. Run Pipeline
    recognizer = ContinuousRecognizer(model_dir)
    
    print("\n--- Pipeline Execution ---")
    emitted = []
    
    for i in range(len(v)):
        emissions = recognizer.feed_frame(v[i])
        for e in emissions:
            print(f"[{i:05d}] DETECTED: {e['label']} (Dist: {e['distance']:.3f}) "
                  f"StartEst: {e['global_start_frame_est']} End: {e['global_end_frame']}")
            emitted.append(e)
            
    print("\n--- Summary ---")
    print(f"Ground Truth Count: {len(meta.get('annotations', []))}")
    print(f"Emitted Count: {len(emitted)}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    syn_file = os.path.join(base_dir, "data", "raw", "synthetic", "syn_001.npz")
    meta_file = os.path.join(base_dir, "data", "raw", "synthetic", "syn_001_meta.json")
    model_dir = os.path.join(base_dir, "models")
    
    test_pipeline_on_synthetic_data(syn_file, meta_file, model_dir)
