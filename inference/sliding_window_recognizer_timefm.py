import os, sys, json
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features_engineering.timefm_extractor import extract_timefm_embeddings

DEFAULT_WINDOW_SIZES = [60, 90, 130, 170, 210, 260, 310, 360]

class SlidingWindowRecognizerTimeFM:
    def __init__(self, model_dir: str, window_sizes: list | None = None, stride: int = 9, confidence_threshold: float = 0.2):
        
        model_path = os.path.join(model_dir, "windowed_classical_timefm_best.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        saved_data = joblib.load(model_path)
        self.clf = saved_data["model"]
        self.model_name = saved_data["model_name"]
        self.classes = saved_data["classes"]
        
        # If the model is LightGBM or XGBoost, the predict_proba logic is the same but classes might be ordered differently
        # Let's ensure self.classes is aligned with the internal predict_proba output.
        
        self.window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
        self.stride = stride
        self.conf_thresh = confidence_threshold

    def recognize(self, frames: np.ndarray, dt: float = 0.02) -> list:
        N = len(frames)
        candidates = []

        feature_batch = []
        meta_batch = []

        for size in self.window_sizes:
            if size > N:
                continue
            for start in range(0, N - size + 1, self.stride):
                window = frames[start : start + size]
                feature_values = extract_timefm_embeddings(window)
                if feature_values is not None:
                    feature_batch.append(feature_values)
                    meta_batch.append((start, size))

        if feature_batch:
            X = np.array(feature_batch)
            probabilities = self.clf.predict_proba(X)
            
            best_window_idx = np.argmax(np.max(probabilities, axis=1))
            best_probs = probabilities[best_window_idx]
            prob_dict = {self.classes[j]: float(best_probs[j]) for j in range(len(self.classes))}
            self.latest_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))
            
            for i, prob in enumerate(probabilities):
                max_index = int(np.argmax(prob))
                highest = float(prob[max_index])
                if highest >= self.conf_thresh:
                    label = self.classes[max_index]
                    if label != "background":
                        start, size = meta_batch[i]
                        
                        prob_dict = {self.classes[j]: float(prob[j]) for j in range(len(self.classes))}
                        sorted_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

                        candidates.append({
                            "label": label,
                            "confidence": highest,
                            "all_probs": sorted_probs,
                            "start": start,
                            "end": start + size - 1,
                            "window_size": size,
                        })

        detections = self.nms(candidates)
        detections.sort(key=lambda detection: detection["start"])
        detections = self.postprocess_bigrams(detections)
        
        return detections

    @staticmethod
    def nms(candidates: list, overlap_thresh: float = 0.1) -> list:
        potentials = sorted(candidates, key=lambda c: (-c["window_size"], -c["confidence"]))
        accepted = []

        while potentials:
            best = potentials.pop(0)
            accepted.append(best)
            remaining = []
            for p in potentials:
                overlap = interval_overlap_ratio(best["start"], best["end"], p["start"], p["end"])
                if overlap <= overlap_thresh:
                    remaining.append(p)
            potentials = remaining

        return accepted

    @staticmethod
    def postprocess_bigrams(detections: list) -> list:
        if not detections:
            return []
            
        final_detections = []
        last_emitted_word = None
        
        for detection in detections:
            label = detection["label"]
            
            if "-" in label:
                parts = label.split("-")
                if len(parts) == 2:
                    word1, word2 = parts
                    
                    if word2 == "nosign":
                        new_d = detection.copy()
                        new_d["label"] = word1
                        final_detections.append(new_d)
                        last_emitted_word = word1
                    elif last_emitted_word == word1:
                        new_d = detection.copy()
                        new_d["label"] = word2
                        final_detections.append(new_d)
                        last_emitted_word = word2
                    else:
                        d1 = detection.copy()
                        d1["label"] = word1
                        d2 = detection.copy()
                        d2["label"] = word2
                        final_detections.extend([d1, d2])
                        last_emitted_word = word2
                else:
                    final_detections.append(detection)
                    last_emitted_word = parts[-1]
            else:
                final_detections.append(detection)
                last_emitted_word = label
                
        return final_detections

def interval_overlap_ratio(start1, end1, start2, end2):
    overlap = min(end1, end2) - max(start1, start2) + 1
    inter = max(0, overlap)
    shorter_sign = min(end1 - start1 + 1, end2 - start2 + 1)
    return inter / shorter_sign if shorter_sign > 0 else 0.0
