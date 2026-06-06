import os, sys, time, json
import numpy as np
import joblib
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.csv_loader import load_sequences
from preprocessing.stream_preprocessor import preprocess_stream
from features_engineering.feature_pipeline import extract_features
import config.pipeline_config as cfg

from sklearn.ensemble import RandomForestClassifier


def augment_sign(values: np.ndarray, rng: np.random.Generator, n_augments: int = 5):
    length = len(values)
    for i in range(n_augments):
        ratio = rng.uniform(0.7, 1.0)
        window = max(5, int(length * ratio))
        max_start = length- window
        start = rng.integers(0, max_start + 1) if max_start > 0 else 0
        yield values[start:start+window], "positive"

    for i in range(2):
        pad_before = rng.integers(3, 12)
        pad_after  = rng.integers(3, 12)
        idle_frame = values[0]
        before = np.tile(idle_frame, (pad_before, 1))
        after  = np.tile(values[-1], (pad_after, 1))
        yield np.vstack([before, values, after]), "positive"

    for i in range(3):
        ratio = rng.uniform(0.20, 0.50)
        window = max(5, int(length * ratio))
        max_start = length - window
        start = rng.integers(0, max_start + 1) if max_start > 0 else 0
        yield values[start:start+window], "background"


def build_dataset(csv_path: str, augment: bool = True):
    sequences = load_sequences(csv_path)
    print(f"  Loaded {len(sequences)} sequences from {os.path.basename(csv_path)}")

    rng = np.random.default_rng(42)
    X_list, y_list = [], []
    skipped = 0

    for sign in sequences:
        frames = sign["frames"]
        label = sign["label"]

        t, values = preprocess_stream(frames[:, 0], frames[:, 1:],disabled_groups=cfg.DISABLED_FEATURE_GROUPS)

        feature_values = extract_features(values, dt=1.0 / cfg.TARGET_SAMPLE_HZ)
        if feature_values is None:
            skipped += 1
            continue
        X_list.append(feature_values)
        y_list.append(label)

        if augment:
            for augmented_values, augmented_type in augment_sign(values, rng, n_augments=5):
                feature_values_aug = extract_features(augmented_values, dt=1.0 / cfg.TARGET_SAMPLE_HZ)
                if feature_values_aug is not None:
                    X_list.append(feature_values_aug)
                    y_list.append(label if augmented_type == "positive" else "background")

    if skipped:
        print(f"  Skipped {skipped} too-short sequences")

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"  Feature matrix: {X.shape}  ({len(set(y))} classes, {len(y)} samples)")
    return X, y


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(base_dir, "data", "processed_csv", "train.csv")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("Building training set with augmentation")
    X_train, y_train = build_dataset(train_csv, augment=True)

    print("\nTraining Random-Forest classifier")
    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=1, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42,)
    clf.fit(X_train, y_train)
    print(f"Training finished in {time.time() - t0:.1f}s")

    train_acc = clf.score(X_train, y_train)
    print(f"In-sample accuracy: {train_acc:.1%}")

    model_path = os.path.join(model_dir, "windowed_rf.joblib")
    joblib.dump(clf, model_path)
    
    label_path = os.path.join(model_dir, "rf_classes.json")
    with open(label_path, "w") as f:
        json.dump(list(clf.classes_), f)
    

if __name__ == "__main__":
    main()
