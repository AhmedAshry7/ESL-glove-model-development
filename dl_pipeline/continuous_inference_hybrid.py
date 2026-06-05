import os
import sys
import json
import torch
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.stream_preprocessor import preprocess_stream
from dl_pipeline.model import SignLanguageTransformer
from recognition.latent_sdtw import LatentSDTWEngine

def _extract_features(seq):
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    acc = np.zeros_like(seq)
    acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=-1)

def run_hybrid_inference():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'data', 'continuous_test.json')
    models_dir = os.path.join(base_dir, 'models', 'dl_models')
    
    model_path = os.path.join(models_dir, 'best_model.pth')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    label_map_path = os.path.join(models_dir, 'label_map.pkl')
    templates_path = os.path.join(models_dir, 'latent_templates.pkl')
    
    if not os.path.exists(templates_path):
        print("Latent templates not found. Run extract_latent_templates.py first.")
        return
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Scaler and Label Map
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_map_path, 'rb') as f:
        label_map = pickle.load(f)
        
    num_classes = len(label_map)
    
    # Load Latent Templates
    with open(templates_path, 'rb') as f:
        latent_templates = pickle.load(f)
        
    # Load Model
    model = SignLanguageTransformer(input_size=168, num_classes=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)
    state_dict.pop('pos_encoder.pe', None)
    model.load_state_dict(state_dict, strict=False)
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
    
    _, processed_features = preprocess_stream(timestamps, raw_feats)
    print(f"Loaded continuous stream with {len(processed_features)} frames.")
    print(f"Ground Truth sequence: {ground_truth}\n")
    
    feats = _extract_features(processed_features)
    scaled_feats = scaler.transform(feats)
    
    inputs = torch.tensor(scaled_feats, dtype=torch.float32).unsqueeze(0).to(device)
    
    print("Extracting Latent Stream using sliding window...")
    latent_stream_list = []
    window_size = 64
    
    # Pad inputs so we can center the window
    pad_len = window_size // 2
    # inputs shape: (1, seq_len, 168)
    padded_inputs = torch.nn.functional.pad(inputs.transpose(1, 2), (pad_len, pad_len), "replicate").transpose(1, 2)
    
    with torch.no_grad():
        for t in range(inputs.shape[1]):
            window = padded_inputs[:, t:t+window_size, :]
            # shape (1, 64, 128)
            latent_window = model.extract_features(window)
            # Take the center frame's embedding
            latent_frame = latent_window[0, pad_len, :].cpu().numpy()
            latent_stream_list.append(latent_frame)
            
    latent_stream_np = np.array(latent_stream_list)
    print("Running Latent SDTW...")
    sdtw = LatentSDTWEngine(templates=latent_templates, default_threshold=0.25)
    
    all_detections = []
    for t in range(len(latent_stream_np)):
        detections = sdtw.feed(latent_stream_np[t], t)
        all_detections.extend(detections)
        
    # NMS (we need a custom NMS for latent embeddings, but we can just use simple overlap NMS)
    print("Running NMS...")
    # The current `perform_nms` in `nms.py` calls `compute_static_dtw` using raw features.
    # Since we are latent, we can just do simple IoU based NMS.
    filtered = []
    all_detections.sort(key=lambda x: x['distance'])
    
    for det in all_detections:
        s1, e1 = det['global_start_frame_est'], det['global_end_frame']
        overlap = False
        for f_det in filtered:
            s2, e2 = f_det['global_start_frame_est'], f_det['global_end_frame']
            # Intersection over Union
            inter = max(0, min(e1, e2) - max(s1, s2))
            union = (e1 - s1) + (e2 - s2) - inter
            if inter / union > 0.3: # 30% overlap threshold
                overlap = True
                break
        if not overlap:
            filtered.append(det)
            
    filtered.sort(key=lambda x: x['global_start_frame_est'])
    
    print("\n--- Final Spotted Sequence ---")
    predictions = [d['label'] for d in filtered]
    print(" -> ".join(predictions))
    
    print("\n--- Ground Truth ---")
    print(ground_truth)
    
    # Print individual confidences
    print("\nDetections:")
    for d in filtered:
        print(f"[{d['global_start_frame_est']:04d} -> {d['global_end_frame']:04d}] {d['label']} (Dist: {d['distance']:.3f})")

if __name__ == '__main__':
    run_hybrid_inference()
