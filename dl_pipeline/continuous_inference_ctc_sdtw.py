import os
import sys
import json
import torch
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.pipeline_config as cfg
from preprocessing.stream_preprocessor import preprocess_stream, normalize_frames
from dl_pipeline.model_ctc import SignLanguageCTCModel
from recognition.sdtw_engine import compute_static_dtw
from features_engineering.template_manager import TemplateManager

def _extract_features(seq):
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    acc = np.zeros_like(seq)
    acc[2:] = vel[2:] - vel[1:-1]
    return np.concatenate([seq, vel, acc], axis=-1)

def run_ctc_sdtw_inference():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'data', 'continuous_test.json')
    models_dir = os.path.join(base_dir, 'models', 'dl_models')
    sdtw_model_dir = os.path.join(base_dir, 'models')
    
    model_path = os.path.join(models_dir, 'best_model_ctc.pth')
    scaler_path = os.path.join(models_dir, 'scaler_ctc.pkl')
    label_map_path = os.path.join(models_dir, 'label_map_ctc.pkl')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Scaler and Label Map for CTC
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_map_path, 'rb') as f:
        label_map = pickle.load(f)
        
    num_classes = len(label_map)
    BLANK_IDX = 0 
    
    # Load Proper SDTW Dependencies
    print("Loading proper SDTW Templates and Weights...")
    template_manager = TemplateManager(os.path.join(sdtw_model_dir, 'templates'))
    templates = template_manager.get_templates()
    
    stats_path = os.path.join(sdtw_model_dir, "templates/normalization_stats.npz")
    if os.path.exists(stats_path):
        data = np.load(stats_path)
        norm_stats = {'mean': data['mean'], 'std': data['std']}
    else:
        norm_stats = None
        print("WARNING: normalization_stats.npz not found.")
        
    weights = cfg.CHANNEL_WEIGHTS
    
    # Load CTC Model
    model = SignLanguageCTCModel(input_size=168, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load Continuous Data
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    ground_truth = data.get("label", "Unknown")
    frames_raw = data.get("frames", [])
    
    frames_array = np.array(frames_raw)
    timestamps = frames_array[:, 0]
    raw_feats = frames_array[:, 1:]
    
    # Preprocess
    _, processed_features = preprocess_stream(timestamps, raw_feats)
    print(f"Loaded continuous stream with {len(processed_features)} frames.")
    
    # Normalizing raw features for SDTW using standard stats
    if norm_stats is not None:
        sdtw_features = normalize_frames(processed_features, norm_stats)
    else:
        sdtw_features = processed_features
    
    # Extract CTC Features
    feats = _extract_features(processed_features)
    scaled_feats = scaler.transform(feats)
    inputs = torch.tensor(scaled_feats, dtype=torch.float32).unsqueeze(0).to(device)
    
    print("\nRunning CTC Gesture Spotter...")
    with torch.no_grad():
        outputs = model(inputs) # (batch, seq_len, num_classes + 1)
        probs = torch.softmax(outputs, dim=2)[0] # (seq_len, num_classes + 1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        
    # Segment extraction
    segments = []
    in_segment = False
    start_idx = 0
    
    for i, p in enumerate(preds):
        if p != BLANK_IDX and not in_segment:
            in_segment = True
            start_idx = i
        elif p == BLANK_IDX and in_segment:
            in_segment = False
            if (i - start_idx) >= 1:
                pad = 15
                seg_start = max(0, start_idx - pad)
                seg_end = min(len(preds), i + pad)
                segments.append((seg_start, seg_end))
                
    if in_segment and (len(preds) - start_idx) >= 1:
        pad = 15
        seg_start = max(0, start_idx - pad)
        seg_end = min(len(preds), len(preds) + pad)
        segments.append((seg_start, seg_end))
        
    print(f"Spotted {len(segments)} potential gestures.")
    
    final_sequence = []
    
    print("\nRunning SDTW Classification (Proper Weights & Norms)...")
    for start, end in segments:
        segment_raw = sdtw_features[start:end]
        
        best_label = None
        best_dist = float('inf')
        
        for label, tmpls in templates.items():
            for tmpl in tmpls:
                dist = compute_static_dtw(segment_raw, tmpl, weights)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label
                    
        print(f"Frames [{start:04d} -> {end:04d}]: SDTW Classified as '{best_label}' (Dist: {best_dist:.3f})")
        final_sequence.append(best_label)
        
    print("\n--- Final Spotted Sequence ---")
    print(" -> ".join(final_sequence))
    print("\n--- Ground Truth ---")
    print(ground_truth)

if __name__ == '__main__':
    run_ctc_sdtw_inference()
