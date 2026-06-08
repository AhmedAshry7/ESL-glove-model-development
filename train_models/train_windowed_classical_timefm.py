import os
import sys
import time
import json
import numpy as np
import joblib
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.csv_loader import load_sequences
from preprocessing.stream_preprocessor import preprocess_stream
from features_engineering.timefm_extractor import extract_timefm_embeddings, extract_timefm_embeddings_batch
import config.pipeline_config as cfg

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score

def augment_sign(values: np.ndarray, rng: np.random.Generator, n_augments: int = 5):
    length = len(values)
    for i in range(n_augments):
        ratio = rng.uniform(0.7, 1.0)
        window = max(5, int(length * ratio))
        max_start = length - window
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
    print(f"Loaded {len(sequences)} sequences from {os.path.basename(csv_path)}")

    rng = np.random.default_rng(42)
    X_list, y_list = [], []
    skipped = 0
    
    all_values = []
    all_labels = []

    for idx, sign in enumerate(sequences):
        frames = sign["frames"]
        label = sign["label"]

        t, values = preprocess_stream(frames[:, 0], frames[:, 1:], disabled_groups=cfg.DISABLED_FEATURE_GROUPS)

        all_values.append(values)
        all_labels.append(label)

        if augment:
            for augmented_values, augmented_type in augment_sign(values, rng, n_augments=3): # Reduced augments to save time for testing
                all_values.append(augmented_values)
                all_labels.append(label if augmented_type == "positive" else "background")

    batch_size = 64
    total_seqs = len(all_values)
    print(f"Extracting embeddings for {total_seqs} sequences in batches of {batch_size}...")
    
    for i in range(0, total_seqs, batch_size):
        batch_vals = all_values[i:i+batch_size]
        batch_lbls = all_labels[i:i+batch_size]
        
        extracted_batch = extract_timefm_embeddings_batch(batch_vals)
        
        for feature_values, lbl in zip(extracted_batch, batch_lbls):
            if feature_values is not None:
                X_list.append(feature_values)
                y_list.append(lbl)
            else:
                skipped += 1
                
        if i > 0 and (i % (batch_size * 5) == 0 or i + batch_size >= total_seqs):
            print(f"Extracted {min(i + batch_size, total_seqs)}/{total_seqs} sequences...")

    if skipped:
        print(f"Skipped {skipped} too-short sequences")

    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"Feature matrix: {X.shape} ({len(set(y))} classes, {len(y)} samples)")
    return X, y

def train_and_evaluate_models(X_train, y_train):
    print("\n--- Training Classical Models ---")
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        # XGBoost requires numeric labels, we'll encode them first
    }
    
    # Label encoding for XGBoost and overall convenience
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    models["XGBoost"] = xgb.XGBClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1, eval_metric="mlogloss", tree_method="hist", device="cuda")
    
    results = {}
    best_acc = 0.0
    best_model_name = ""
    best_model = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        
        if name in ["XGBoost", "LightGBM"]:
            model.fit(X_train, y_train_encoded)
            preds = model.predict(X_train)
            acc = accuracy_score(y_train_encoded, preds)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            acc = accuracy_score(y_train, preds)
            
        train_time = time.time() - t0
        print(f"{name} trained in {train_time:.1f}s | In-sample accuracy: {acc:.1%}")
        
        results[name] = {"accuracy": acc, "train_time": train_time}
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model

    print(f"\nBest Model: {best_model_name} with {best_acc:.1%} accuracy.")
    return best_model, best_model_name, le, results

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(base_dir, "data", "processed_csv", "train.csv")
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    print("Building training set with TimeFM embeddings (this may take a while)...")
    X_train, y_train = build_dataset(train_csv, augment=True)

    best_model, best_model_name, label_encoder, results = train_and_evaluate_models(X_train, y_train)

    # Save the best model
    model_path = os.path.join(model_dir, "windowed_classical_timefm_best.joblib")
    
    # Pack model and metadata
    saved_data = {
        "model_name": best_model_name,
        "model": best_model,
        "label_encoder": label_encoder if best_model_name in ["XGBoost", "LightGBM"] else None,
        "classes": list(label_encoder.classes_)
    }
    
    joblib.dump(saved_data, model_path)
    print(f"\nSaved {best_model_name} model to {model_path}")
    
    label_path = os.path.join(model_dir, "timefm_classes.json")
    with open(label_path, "w") as f:
        json.dump(list(label_encoder.classes_), f)

if __name__ == "__main__":
    main()
