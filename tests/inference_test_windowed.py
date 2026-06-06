import sys, os, json
import numpy as np
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.pipeline_config as cfg
from preprocessing.stream_preprocessor import preprocess_stream
from inference.sliding_window_recognizer import SlidingWindowRecognizer


def test_continuous_files(json_files: list, model_dir: str):
    recognizer = SlidingWindowRecognizer(model_dir)

    total_sequence = 0
    correct = 0
    class_correct, class_total = {}, {}
    confusion = []

    for json_file in json_files:
        split_name = os.path.basename(json_file).replace(".json", "")
        print(f"Assessing: {split_name}")

        with open(json_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        sequences = [loaded] if isinstance(loaded, dict) else loaded
        total_sequence += len(sequences)

        for sign in sequences:
            label = sign["label"]
            raw_frames = np.array(sign["frames"])

            labels = [l.strip() for l in label.replace(",", "-").split("-") if l.strip() and l.strip() != "nosign"]

            for l in labels:
                class_total[l] = class_total.get(l, 0) + 1

            t, values = preprocess_stream(raw_frames[:, 0], raw_frames[:, 1:], disabled_groups=cfg.DISABLED_FEATURE_GROUPS)

            detections = recognizer.recognize(values, dt=1.0 / cfg.TARGET_SAMPLE_HZ)
            pred_labels = [d["label"] for d in detections]
            
            if pred_labels == labels:
                correct += 1
            else:
                confusion.append((str(labels), str(pred_labels)))

            label_count = Counter(labels)
            prediction_count = Counter(pred_labels)
            for l, c in (label_count & prediction_count).items():
                class_correct[l] = class_correct.get(l, 0) + c

            labels_d = ", ".join(labels)
            predictions_d = ", ".join(pred_labels) if pred_labels else "NO DETECTION"
            det_info = "  ".join(f"[{d['label']} {d['confidence']:.2f} f{d['start']}-{d['end']}]" for d in detections) if detections else ""
            print(f"Label:   [{labels_d}]")
            print(f"Prediction: [{predictions_d}]")
            if det_info:
                print(f"DETS: {det_info}")
            print()

    seq_acc = correct / total_sequence * 100 if total_sequence else 0

    print(f"Sequence Exact-Match Accuracy: {seq_acc:.1f}% ({correct}/{total_sequence})\n")
    print(f"{'Class':<20} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    for label in sorted(class_total.keys()):
        correct = class_correct.get(label, 0)
        total = class_total[label]
        percentage = correct / total * 100 if t else 0
        print(f"  {label:<20} {correct:>5}/{total:<5} {percentage:>6.0f}%")

    if confusion:
        print(f"\n  Most Common Mismatches:")
        for (labels, predictions), count in Counter(confusion).most_common(10):
            print(f"label: {labels} prediction: {predictions}  ({count}x)")

    return seq_acc


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")

    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    files_to_test = []
    inference_dir = os.path.join(base_dir, "data", "inference_data")

    if arg == "all":
        import glob
        files_to_test = glob.glob(os.path.join(inference_dir, "*.json"))
    elif arg.endswith(".json"):
        if os.path.exists(arg):
            files_to_test = [arg]
        elif os.path.exists(os.path.join(inference_dir, arg)):
            files_to_test = [os.path.join(inference_dir, arg)]
    elif os.path.exists(os.path.join(inference_dir, f"{arg}.json")):
        files_to_test = [os.path.join(inference_dir, f"{arg}.json")]
    else:
        files_to_test = [os.path.join(base_dir, "data", f"{arg}.json")]

    if not files_to_test:
        print("No files found to test.")
        sys.exit(1)

    test_continuous_files(files_to_test, model_dir)
