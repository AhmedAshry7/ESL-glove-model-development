import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.pipeline_config as cfg
from features_engineering.prefilter_features import extract_prefilter_dataset, load_norm_stats

def train_prefilter(train_csv_path: str, model_dir: str, disabled_groups: list = []):
    norm_stats = load_norm_stats(model_dir)
    if norm_stats is None:
        print("normalization stats not found.")

    X, Y = extract_prefilter_dataset(train_csv_path, disabled_groups, norm_stats=norm_stats)

    if len(X) == 0:
        print("No training data found. Make sure you have data in train_csv_path.")
        return


    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    clf.fit(X, Y)

    y_pred = clf.predict(X)
    acc = np.mean(y_pred == Y)
    print(f"Training Accuracy (Top-1): {acc*100:.2f}%")
    model_path = os.path.join(model_dir, "prefilter_rf.joblib")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed_csv", "train.csv")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    train_prefilter(train_csv_path, model_dir, disabled_groups=cfg.DISABLED_FEATURE_GROUPS)
