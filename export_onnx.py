import os
import joblib
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features_engineering.feature_pipeline import feature_vector_length

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
except ImportError:
    print("skl2onnx not installed. Please install it first.")
    sys.exit(1)

def main():
    model_path = os.path.join("models", "windowed_rf.joblib")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        sys.exit(1)

    clf = joblib.load(model_path)
    
    n_features = feature_vector_length()

    initial_type = [('float_input', FloatTensorType([None, n_features]))]

    print("Converting model to ONNX...")
    # 'zipmap': False makes the output probability a tensor rather than a list of dictionaries,
    # which is often much easier to parse in Flutter/Dart with onnxruntime.
    options = {id(clf): {'zipmap': False}}
    onx = convert_sklearn(clf, initial_types=initial_type, options=options, target_opset=9)

    out_path = os.path.join("models", "windowed_rf.onnx")
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    
    print(f"ONNX model saved to {out_path}")

    classes = clf.classes_
    classes_path = os.path.join("models", "classes.json")
    import json
    with open(classes_path, "w") as f:
        json.dump(list(classes), f, indent=4)
    print(f"Classes saved to {classes_path}")

if __name__ == "__main__":
    main()
