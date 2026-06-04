import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.stream_preprocessor import preprocess_stream
from inference.continuous_recognizer import ContinuousRecognizer
from data.csv_loader import load_sequences_from_csv

def test_pipeline_on_csv(csv_file: str, model_dir: str):
    print(f"Testing pipeline on {csv_file}")
    
    # 1. Load Data
    sequences = load_sequences_from_csv(csv_file)
    print(f"Loaded {len(sequences)} sequences from CSV.")
    
    # 2. Run Pipeline
    recognizer = ContinuousRecognizer(model_dir)
    
    # Extract mean from the loaded recognizer to pad correctly
    if recognizer.norm_stats is not None:
        mean_pose = recognizer.norm_stats['mean']
    else:
        mean_pose = np.zeros(56)
    
    total_sequences = len(sequences)
    correct_predictions = 0
    
    print("\n--- Pipeline Execution ---")
    
    for seq_idx, sign in enumerate(sequences):
        # Reset the recognizer state for each independent sequence
        recognizer._reset_state()
        
        ground_truth = sign['label']
        raw_frames = sign['frames']
        
        t, v = preprocess_stream(raw_frames[:, 0], raw_frames[:, 1:], disabled_groups=[])
        
        # Pad with 20 idle frames (using the dataset mean) so the ActivityMonitor detects rest and flushes
        idle_frames = np.tile(mean_pose, (20, 1))
        v_padded = np.vstack((v, idle_frames))
        
        emissions = []
        for i in range(len(v_padded)):
            result = recognizer.feed_frame(v_padded[i])
            if result:
                emissions.extend(result)
                
        # Get the most confident emission for this sequence
        if emissions:
            best_emission = min(emissions, key=lambda e: e['distance'])
            predicted = best_emission['label']
            dist = best_emission['distance']
            
            if predicted == ground_truth:
                correct_predictions += 1
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"
                
            print(f"[{seq_idx+1}/{total_sequences}] GT: {ground_truth:<15} PRED: {predicted:<15} (Dist: {dist:.3f}) {status}")
        else:
            if ground_truth == "no_sign":
                correct_predictions += 1
                print(f"[{seq_idx+1}/{total_sequences}] GT: {ground_truth:<15} PRED: NO DETECTION                      ✅ CORRECT (True Negative)")
            else:
                print(f"[{seq_idx+1}/{total_sequences}] GT: {ground_truth:<15} PRED: NO DETECTION                      ❌ MISS")
            
    print("\n--- Summary ---")
    print(f"Total Sequences: {total_sequences}")
    accuracy = (correct_predictions / total_sequences) * 100 if total_sequences > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_sequences})")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_csv = os.path.join(base_dir, "data", "processed_csv", "test.csv")
    model_dir = os.path.join(base_dir, "models")
    
    test_pipeline_on_csv(test_csv, model_dir)
