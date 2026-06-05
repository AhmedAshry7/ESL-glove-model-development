import os
import sys
import json
import torch
import numpy as np
import pickle
from scipy.interpolate import interp1d

# Add root directory to path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.stream_preprocessor import preprocess_stream
from dl_pipeline.model import SignLanguageTransformer

def _interpolate_sequence(seq, target_len=64):
    T, F = seq.shape
    if T == target_len:
        return seq
    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, target_len)
    interpolator = interp1d(old_indices, seq, axis=0, kind='linear', fill_value="extrapolate")
    return interpolator(new_indices)

def _extract_features(seq):
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    acc = np.zeros_like(seq)
    acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=-1)

def run_continuous_inference():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'data', 'continuous_test.json')
    models_dir = os.path.join(base_dir, 'models', 'dl_models')
    
    model_path = os.path.join(models_dir, 'best_model.pth')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    label_map_path = os.path.join(models_dir, 'label_map.pkl')
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
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
    max_seq_len = 64
    
    # Load Model
    model = SignLanguageTransformer(input_size=input_size, num_classes=num_classes).to(device)
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
    
    # Gesture Spotting Parameters
    ENERGY_THRESHOLD_START = 5.0
    ENERGY_THRESHOLD_STOP = 3.0
    MIN_SIGN_FRAMES = 20
    MAX_SILENCE_FRAMES = 15 # Increased to avoid cutting signs in half during brief pauses
    
    predictions = []
    
    state = "IDLE" # can be IDLE or SIGNING
    buffer = []
    silence_counter = 0
    energy_buffer = [] # To keep smoothed energy
    
    prev_features = None
    
    print("Starting Gesture Spotting Inference...\n")
    
    with torch.no_grad():
        for i, features in enumerate(processed_features):
            features = features.astype(np.float32)
            
            if prev_features is None:
                energy = 0
            else:
                energy = np.sum(np.abs(features - prev_features))
                
            energy_buffer.append(energy)
            if len(energy_buffer) > 5: # 5-frame moving average
                energy_buffer.pop(0)
                
            smoothed_energy = np.mean(energy_buffer)
                
            prev_features = features
            
            # State Machine
            if state == "IDLE":
                if smoothed_energy > ENERGY_THRESHOLD_START:
                    state = "SIGNING"
                    buffer = [features]
                    silence_counter = 0
            elif state == "SIGNING":
                buffer.append(features)
                if smoothed_energy < ENERGY_THRESHOLD_STOP:
                    silence_counter += 1
                else:
                    silence_counter = 0
                    
                if silence_counter >= MAX_SILENCE_FRAMES:
                    # Sign has ended
                    state = "IDLE"
                    
                    # Remove the silence frames from the buffer
                    sign_sequence = buffer[:-MAX_SILENCE_FRAMES]
                    
                    if len(sign_sequence) >= MIN_SIGN_FRAMES:
                        # Process isolated sign
                        seq = np.array(sign_sequence)
                        interpolated = _interpolate_sequence(seq, target_len=max_seq_len)
                        feat = _extract_features(interpolated)
                        
                        flat_feat = feat.reshape(-1, feat.shape[-1])
                        scaled_feat = scaler.transform(flat_feat)
                        scaled_seq = scaled_feat.reshape(1, max_seq_len, -1).astype(np.float32)
                        
                        inputs = torch.tensor(scaled_seq).to(device)
                        outputs = model(inputs)
                        probs = torch.softmax(outputs, dim=1)[0]
                        
                        max_prob, predicted_idx = torch.max(probs, 0)
                        pred_label = inverse_label_map[predicted_idx.item()]
                        conf = max_prob.item()
                        
                        print(f"Frames [{i - len(buffer):04d} -> {i:04d}]: Detected '{pred_label}' (Conf: {conf:.2f})")
                        predictions.append(pred_label)
                        
    print("\n--- Final Spotted Sequence ---")
    print(" -> ".join(predictions))
    print("\n--- Ground Truth ---")
    print(ground_truth)

if __name__ == '__main__':
    run_continuous_inference()
