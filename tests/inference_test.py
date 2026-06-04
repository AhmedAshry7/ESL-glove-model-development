import sys
import os
import json
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.pipeline_config as cfg
from preprocessing.stream_preprocessor import preprocess_stream, normalize_frames
from inference.continuous_recognizer import ContinuousRecognizer

def test_continuous_pipeline_on_json(json_file: str, model_dir: str, split_name: str = "continuous_test"):
    print(f"Evaluating continuous inference on {split_name.upper()} set: {json_file}")
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found.")
        sys.exit(1)
        
    with open(json_file, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    # Safely handle both a single continuous JSON dictionary AND a list of JSON objects
    if isinstance(loaded_data, dict):
        sequences = [loaded_data]
    elif isinstance(loaded_data, list):
        sequences = loaded_data
    else:
        print(f"Error: Unexpected JSON structure in {json_file}. Expected object or list.")
        sys.exit(1)
    
    print(f"Loaded {len(sequences)} continuous sequence(s) from JSON.\n")
    
    recognizer = ContinuousRecognizer(model_dir)
    if recognizer.norm_stats is not None:
        mean_pose = recognizer.norm_stats['mean']
    else:
        mean_pose = np.zeros(56)
    
    total_sequences = len(sequences)
    correct_sequences = 0
    
    class_correct = {}
    class_total = {}
    confusion_sequences = []
    
    print("Per-Sequence Continuous Results\n")
    
    for sequence_index, sign in enumerate(sequences):
        recognizer._reset_state()
        
        # Now sign is guaranteed to be a dictionary
        ground_truth_str = sign['label']
        raw_frames = np.array(sign['frames'])
        
        # Split comma-separated (or hyphen-separated) labels into a clean list
        # Handles delimiters like: "marhaban,ana" or "marhaban-ana"
        delimiters = [',', '-']
        normalized_str = ground_truth_str
        for d in delimiters:
            normalized_str = normalized_str.replace(d, ',')
            
        gt_labels = [lbl.strip() for lbl in normalized_str.split(',') if lbl.strip()]
        
        for lbl in gt_labels:
            class_total[lbl] = class_total.get(lbl, 0) + 1
        
        # Split timestamps and features (column 0 = timestamp, columns 1+ = features)
        t, v = preprocess_stream(raw_frames[:, 0], raw_frames[:, 1:], disabled_groups=cfg.DISABLED_FEATURE_GROUPS)
       
        # Pad with 40 idle frames so the ActivityMonitor detects rest and flushes remaining valleys
        idle_frames = np.tile(mean_pose, (40, 1))
        v_padded = np.vstack((v, idle_frames))
        
        emissions = []
        for i in range(len(v_padded)):
            result = recognizer.feed_frame(v_padded[i])
            if result:
                emissions.extend(result)
                
        # Extract predicted labels in chronological order of emission
        predicted_labels = [e['label'] for e in emissions]
        
        # Check if full sequence matches exactly in order
        if predicted_labels == gt_labels:
            correct_sequences += 1
            status = "✅"
        else:
            status = "❌"
            confusion_sequences.append((str(gt_labels), str(predicted_labels)))
            
        # Update token-level class correctness using Counter intersection
        gt_counter = Counter(gt_labels)
        pred_counter = Counter(predicted_labels)
        match_counter = gt_counter & pred_counter
        for lbl, match_count in match_counter.items():
            class_correct[lbl] = class_correct.get(lbl, 0) + match_count
            
        gt_display = ", ".join(gt_labels)
        pred_display = ", ".join(predicted_labels) if predicted_labels else "NO DETECTION"
        
        print(f"  [{sequence_index+1:>3}/{total_sequences}] GT:   [{gt_display}]")
        print(f"        → PRED: [{pred_display}] {status}\n")
            
    sequence_accuracy = (correct_sequences / total_sequences) * 100 if total_sequences > 0 else 0
    
    print(f"\n{'─'*60}")
    print(f" {split_name.upper()} SET SUMMARY")
    print(f"{'─'*60}")
    print(f"  Sequence-level Exact Match Accuracy: {sequence_accuracy:.1f}% ({correct_sequences}/{total_sequences})")
    print(f"\n  {'Class':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print(f"  {'─'*47}")
    for label in sorted(class_total.keys()):
        c = class_correct.get(label, 0)
        t = class_total[label]
        pct = (c / t * 100) if t > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {label:<20} {c:>5}/{t:<5} {pct:>6.0f}%  {bar}")
    
    if confusion_sequences:
        print(f"\n  Most Common Sequence Mismatches:")
        error_counts = Counter(confusion_sequences)
        for (gt_seq, pred_seq), count in error_counts.most_common(5):
            print(f"    GT: {gt_seq} → PRED: {pred_seq}  ({count}x)")
            
    print(f"{'─'*60}\n")
    return sequence_accuracy

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    
    argument = sys.argv[1] if len(sys.argv) > 1 else "continuous_test"
    
    if argument.endswith('.json') and os.path.exists(argument):
        json_file = argument
        split_name = os.path.basename(argument).replace('.json', '')
    else:
        json_file = os.path.join(base_dir, "data", f"{argument}.json")
        split_name = argument
        
    test_continuous_pipeline_on_json(json_file, model_dir, split_name=split_name)