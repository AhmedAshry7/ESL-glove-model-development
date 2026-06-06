import sys, os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.pipeline_config as cfg
from data.csv_loader import load_sequences
from preprocessing.stream_preprocessor import preprocess_stream
from features_engineering.feature_pipeline import extract_features
import joblib


def test_pipeline(csv_file: str, model_dir: str, split_name: str = "test"):
    print(f"Evaluating on {split_name.upper()} set: {csv_file}")

    sequences = load_sequences(csv_file)
    print(f"Loaded {len(sequences)} sequences.\n")

    rf_path = os.path.join(model_dir, "windowed_rf.joblib")
    clf = joblib.load(rf_path)

    total = len(sequences)
    correct = 0
    class_correct, class_total = {}, {}
    confusion = []

    for idx, sign in enumerate(sequences):
        label = sign["label"]
        raw = sign["frames"]
        class_total[label] = class_total.get(label, 0) + 1

        t, values = preprocess_stream(raw[:, 0], raw[:, 1:], disabled_groups=cfg.DISABLED_FEATURE_GROUPS)
        feature_values = extract_features(values, dt=1.0 / cfg.TARGET_SAMPLE_HZ)

        if feature_values is None:
            confusion.append((label, "TOO_SHORT"))
            print(f"Label: {label:<20} → Predicted: TOO SHORT ❌")
            continue

        probability = clf.predict_proba(feature_values.reshape(1, -1))[0]
        prediction = clf.classes_[np.argmax(probability)]
        confidence = np.max(probability)

        if prediction == label:
            correct += 1
            class_correct[label] = class_correct.get(label, 0) + 1
        else:
            confusion.append((label, prediction))

        print(f"Label: {label:<20} → Predicted: {prediction:<20} ({confidence:.2f})")

    acc = correct / total * 100 if total else 0.0


    print(f"Overall Accuracy: {acc:.1f}% ({correct}/{total})\n")
    print(f"\n  {'Class':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    for label in sorted(class_total.keys()):
        correct= class_correct.get(label, 0)
        total = class_total[label]
        percentage = correct / total * 100 if total else 0
        print(f"{label:<20} {correct:>5}/{total:<5} {percentage:>6.0f}%")

    if confusion:
        print(f"\n  Most Common Errors:")
        from collections import Counter
        for (label, prediction), count in Counter(confusion).most_common(10):
            print(f"{label}  {prediction}  {count}")

    return acc


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")

    arg = sys.argv[1] if len(sys.argv) > 1 else "test"
    csv_file = os.path.join(base_dir, "data", "processed_csv", f"{arg}.csv")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the pipeline first.")
        sys.exit(1)

    test_pipeline(csv_file, model_dir, split_name=arg)
