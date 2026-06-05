"""
test_windowed_pipeline.py

Evaluate the windowed-RF pipeline on per-sign CSV test/val data.
Each sign is a single recording — we extract features and classify directly.
"""

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
        gt = sign["label"]
        raw = sign["frames"]
        class_total[gt] = class_total.get(gt, 0) + 1

        t, values = preprocess_stream(raw[:, 0], raw[:, 1:], disabled_groups=cfg.DISABLED_FEATURE_GROUPS)
        feature_values = extract_features(values, dt=1.0 / cfg.TARGET_SAMPLE_HZ)

        if feature_values is None:
            confusion.append((gt, "TOO_SHORT"))
            print(f"  [{idx+1:>3}/{total}] GT: {gt:<20} → PRED: TOO SHORT ❌")
            continue

        proba = clf.predict_proba(feature_values.reshape(1, -1))[0]
        pred = clf.classes_[np.argmax(proba)]
        conf = np.max(proba)

        if pred == gt:
            correct += 1
            class_correct[gt] = class_correct.get(gt, 0) + 1
            status = "✅"
        else:
            status = "❌"
            confusion.append((gt, pred))

        print(f"  [{idx+1:>3}/{total}] GT: {gt:<20} → PRED: {pred:<20} ({conf:.2f}) {status}")

    acc = correct / total * 100 if total else 0.0

    print(f"\n{'─'*60}")
    print(f" {split_name.upper()} SET SUMMARY")
    print(f"{'─'*60}")
    print(f"  Overall Accuracy: {acc:.1f}% ({correct}/{total})")
    print(f"\n  {'Class':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print(f"  {'─'*47}")
    for label in sorted(class_total.keys()):
        c = class_correct.get(label, 0)
        t = class_total[label]
        pct = c / t * 100 if t else 0
        bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
        print(f"  {label:<20} {c:>5}/{t:<5} {pct:>6.0f}%  {bar}")

    if confusion:
        print(f"\n  Most Common Errors:")
        from collections import Counter
        for (gt, pred), cnt in Counter(confusion).most_common(10):
            print(f"    {gt} → {pred}  ({cnt}x)")

    print(f"{'─'*60}\n")
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
