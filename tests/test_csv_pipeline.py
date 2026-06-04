import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.pipeline_config as cfg
from preprocessing.stream_preprocessor import preprocess_stream, normalize_frames
from inference.continuous_recognizer import ContinuousRecognizer
from data.csv_loader import load_sequences

def test_pipeline_on_csv(csv_file: str, model_dir: str, split_name: str = "test"):
    print(f"Evaluating on {split_name.upper()} set: {csv_file}")
    
    sequences = load_sequences(csv_file)
    print(f"Loaded {len(sequences)} sequences from CSV.\n")
    
    recognizer = ContinuousRecognizer(model_dir)
    if recognizer.norm_stats is not None:
        mean_pose = recognizer.norm_stats['mean']
    else:
        mean_pose = np.zeros(56)
    
    total_sequences = len(sequences)
    correct_predictions = 0
    
    class_correct = {}
    class_total = {}
    confusion_pairs = []
    
    print("Per-Sequence Results\n")
    
    for sequence_index, sign in enumerate(sequences):
        recognizer._reset_state()
        
        ground_truth = sign['label']
        raw_frames = sign['frames']
        class_total[ground_truth] =class_total.get(ground_truth, 0) + 1
        
        t, v = preprocess_stream(raw_frames[:, 0], raw_frames[:, 1:], disabled_groups=cfg.DISABLED_FEATURE_GROUPS)
       
        # Pad with 40 idle frames (using the dataset mean) so the ActivityMonitor detects rest and flushes
        # Longer signs (120+ frames) need more trailing silence for SDTW valleys to complete
        idle_frames = np.tile(mean_pose, (40, 1))
        v_padded = np.vstack((v, idle_frames))
        emissions = []
        for i in range(len(v_padded)):
            result = recognizer.feed_frame(v_padded[i])
            if result:
                emissions.extend(result)
                
        if emissions:
            best_emission = min(emissions, key=lambda e: e['distance'])
            predicted = best_emission['label']
            dist = best_emission['distance']
            
            if predicted == ground_truth:
                correct_predictions += 1
                class_correct[ground_truth] = class_correct.get(ground_truth, 0) + 1
                status = "✅"
            else:
                status = "❌"
                confusion_pairs.append((ground_truth, predicted, dist))
                
            print(f"  [{sequence_index+1:>3}/{total_sequences}] GT: {ground_truth:<20} → PRED: {predicted:<20} (Dist: {dist:.3f}) {status}")
        else:
            if ground_truth == "no_sign":
                correct_predictions += 1
                class_correct[ground_truth] = class_correct.get(ground_truth, 0) + 1
                print(f"  [{sequence_index+1:>3}/{total_sequences}] GT: {ground_truth:<20} → PRED: NO DETECTION              ✅ (True Negative)")
            else:
                confusion_pairs.append((ground_truth, "NO_DETECTION", 1.0))
                print(f"  [{sequence_index+1:>3}/{total_sequences}] GT: {ground_truth:<20} → PRED: NO DETECTION              ❌ MISS")
            
    accuracy = (correct_predictions / total_sequences) * 100 if total_sequences > 0 else 0
    
    print(f"\n{'─'*60}")
    print(f" {split_name.upper()} SET SUMMARY")
    print(f"{'─'*60}")
    print(f"  Overall Accuracy: {accuracy:.1f}% ({correct_predictions}/{total_sequences})")
    print(f"\n  {'Class':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print(f"  {'─'*47}")
    for label in sorted(class_total.keys()):
        c = class_correct.get(label, 0)
        t = class_total[label]
        pct = (c / t * 100) if t > 0 else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {label:<20} {c:>5}/{t:<5} {pct:>6.0f}%  {bar}")
    
    if confusion_pairs:
        print(f"\n  Most Common Errors:")
        from collections import Counter
        error_counts = Counter((gt, pred) for gt, pred, _ in confusion_pairs)
        for (gt, pred), count in error_counts.most_common(10):
            print(f"    {gt} → {pred}  ({count}x)")
    
    print(f"{'─'*60}\n")
    return accuracy

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    
    # Default test
    arguement = sys.argv[1] if len(sys.argv) > 1 else "test"
    csv_file = os.path.join(base_dir, "data", "processed_csv", f"{arguement}.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the pipeline first.")
        sys.exit(1)
    
    test_pipeline_on_csv(csv_file, model_dir, split_name=arguement)
