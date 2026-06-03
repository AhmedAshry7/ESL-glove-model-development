import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.pipeline_config as cfg
from features_engineering.prefilter_features import extract_prefilter_dataset, load_norm_stats

def train_prefilter(raw_dir: str, model_dir: str, disabled_groups: list = []):
    """
    Train a Random Forest classifier on Option-D rolling features to act as
    the Pre-filter.  Saves the trained model to model_dir/prefilter_rf.joblib.

    NOTE: build_template_db.py must be run first to generate normalization_stats.npz.
    """
    # Load normalization stats produced by build_template_db
    norm_stats = load_norm_stats(model_dir)
    if norm_stats is None:
        print("WARNING: normalization_stats.npz not found in model_dir.")
        print("  Run build_template_db.py first, then re-train the prefilter.")
        print("  Continuing without normalization (results will be suboptimal).")

    print(f"Extracting features from {raw_dir}…")
    X, Y = extract_prefilter_dataset(raw_dir, disabled_groups, norm_stats=norm_stats)

    if len(X) == 0:
        print("No training data found. Make sure you have data in raw_dir.")
        return

    print(f"Training Random Forest on {len(X)} samples with {X.shape[1]} features…")

    # Random Forest optimized for fast inference on mobile
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
    clf.fit(X, Y)

    # Calculate simple training accuracy
    y_pred = clf.predict(X)
    acc = np.mean(y_pred == Y)
    print(f"Training Accuracy (Top-1): {acc*100:.2f}%")

    # Save the model
    model_path = os.path.join(model_dir, "prefilter_rf.joblib")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    raw_dir   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    train_prefilter(raw_dir, model_dir, disabled_groups=cfg.DISABLED_FEATURE_GROUPS)

