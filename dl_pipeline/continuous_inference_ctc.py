import os
import sys
import json
import torch
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.stream_preprocessor import preprocess_stream
from dl_pipeline.model_ctc import SignLanguageCTCModel

def _extract_features(seq):
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    acc = np.zeros_like(seq)
    acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=-1)

def decode_greedy(log_probs, inverse_label_map):
    # log_probs shape: (1, T, C)
    preds = log_probs.argmax(dim=-1)[0] # (T,)
    seq = []
    prev_idx = -1
    for t in range(preds.shape[0]):
        idx = preds[t].item()
        if idx != 0 and idx != prev_idx:
            seq.append(inverse_label_map[idx])
        prev_idx = idx
    return seq

def run_continuous_inference_ctc():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'data', 'continuous_test.json')
    models_dir = os.path.join(base_dir, 'models', 'dl_models')
    
    model_path = os.path.join(models_dir, 'best_model_ctc.pth')
    scaler_path = os.path.join(models_dir, 'scaler_ctc.pkl')
    label_map_path = os.path.join(models_dir, 'label_map_ctc.pkl')
    
    if not os.path.exists(json_path) or not os.path.exists(model_path):
        print(f"Files not found. Please train the CTC model first.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Scaler and Label Map
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_map_path, 'rb') as f:
        label_map = pickle.load(f)
        
    inverse_label_map = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    input_size = 56 * 3
    
    # Load Model
    model = SignLanguageCTCModel(input_size=input_size, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Continuous Data
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    ground_truth = data.get("label", "Unknown")
    frames_raw = data.get("frames", [])
    
    # Process the raw frames just like training data
    frames_array = np.array(frames_raw)
    timestamps = frames_array[:, 0]
    raw_feats = frames_array[:, 1:]
    
    # Preprocess: normalize quaternions and resample to uniform 50Hz
    _, processed_features = preprocess_stream(timestamps, raw_feats)
    
    print(f"Loaded continuous stream with {len(processed_features)} processed frames.")
    print(f"Ground Truth sequence: {ground_truth}\n")
    
    print("Starting CTC Inference...\n")
    
    # Feed entire stream at once (CTC can handle this)
    feats = _extract_features(processed_features)
    scaled_feats = scaler.transform(feats)
    
    inputs = torch.tensor(scaled_feats, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        log_probs = model(inputs) # (1, T, C)
        
        predictions = decode_greedy(log_probs, inverse_label_map)
        
    print("\n--- Final Spotted Sequence ---")
    print(" -> ".join(predictions))
    print("\n--- Ground Truth ---")
    print(ground_truth)

if __name__ == '__main__':
    run_continuous_inference_ctc()
