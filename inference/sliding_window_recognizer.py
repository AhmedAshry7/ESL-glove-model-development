"""
sliding_window_recognizer.py

Replaces the ActivityMonitor + SDTW architecture with a multi-scale
sliding-window approach:

  1. Slide windows of several sizes across the stream.
  2. For each window: extract features → classify with RF → get probabilities.
  3. Apply greedy NMS to select the best non-overlapping detections.

This eliminates the need for an activity-monitor to segment continuous
recordings — the RF confidence at each window position *is* the
segmentation signal.
"""

import os, sys, json
import numpy as np
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features_engineering.feature_pipeline import extract_features

# ── Default multi-scale window sizes (frames @ 50 Hz) ──
# Covers the observed sign-length range: 64-frame shortest to 379-frame longest.
DEFAULT_WINDOW_SIZES = [60, 90, 130, 170, 210, 260, 310, 360]
DEFAULT_STRIDE = 9
DEFAULT_CONFIDENCE_THRESHOLD = 0.20


class SlidingWindowRecognizer:
    """Multi-scale sliding-window sign recognizer."""

    def __init__(self, model_dir: str,
                 window_sizes: list | None = None,
                 stride: int = DEFAULT_STRIDE,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):

        rf_path = os.path.join(model_dir, "windowed_rf.joblib")
        if not os.path.exists(rf_path):
            raise FileNotFoundError(f"RF model not found: {rf_path}")
        self.clf = joblib.load(rf_path)
        self.classes = list(self.clf.classes_)

        self.window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
        self.stride = stride
        self.conf_thresh = confidence_threshold

    # ──────────────────────────────────────────────────────────────────────
    def recognize(self, frames: np.ndarray, dt: float = 0.02) -> list:
        """Run multi-scale sliding-window recognition.

        Parameters
        ----------
        frames : (N, 56) resampled sensor data (raw, NOT normalised).
        dt     : sample period in seconds (default 0.02 = 50 Hz).

        Returns
        -------
        List of detection dicts sorted by start frame:
            [{'label': str, 'confidence': float,
              'start': int, 'end': int, 'window_size': int}, …]
        """
        N = len(frames)
        candidates = []

        for win in self.window_sizes:
            if win > N:
                continue
            for start in range(0, N - win + 1, self.stride):
                window = frames[start : start + win]
                fv = extract_features(window, dt)
                if fv is None:
                    continue

                proba = self.clf.predict_proba(fv.reshape(1, -1))[0]
                best_idx = int(np.argmax(proba))
                best_conf = float(proba[best_idx])

                if best_conf >= 0.05:
                    label = self.classes[best_idx]
                    if label != "background":
                        candidates.append({
                            "label": label,
                            "confidence": best_conf,
                            "start": start,
                            "end": start + win - 1,
                            "window_size": win,
                        })

        # ── Greedy NMS ──────────────────────────────────────────────────
        detections = self._nms(candidates)
        detections.sort(key=lambda d: d["start"])
        
        # ── Bigram Post-processing ──────────────────────────────────────
        detections = self._postprocess_bigrams(detections)
        
        return detections

    @staticmethod
    def _nms(candidates: list, overlap_thresh: float = 0.1) -> list:
        """Greedy non-maximum suppression on 1-D intervals.

        Picks the highest-confidence candidate, removes all candidates
        that overlap with it by more than *overlap_thresh* (Intersection over Minimum),
        and repeats.
        """
        if not candidates:
            return []

        cands = sorted(candidates, key=lambda c: -c["confidence"])
        accepted = []

        while cands:
            best = cands.pop(0)
            accepted.append(best)
            remaining = []
            for c in cands:
                overlap = _interval_overlap_ratio(best["start"], best["end"],
                                                  c["start"], c["end"])
                if overlap <= overlap_thresh:
                    remaining.append(c)
            cands = remaining

        return accepted

    @staticmethod
    def _postprocess_bigrams(detections: list) -> list:
        """
        Process bigram labels (e.g. 'marhaban-ana').
        If the first word matches the previously emitted word, suppress it
        and emit only the second word. Otherwise, emit both words.
        """
        if not detections:
            return []
            
        final_detections = []
        last_emitted_word = None
        
        for d in detections:
            label = d["label"]
            
            if "-" in label:
                parts = label.split("-")
                if len(parts) == 2:
                    w1, w2 = parts
                    
                    if w2 == "nosign":
                        # Handle isolated sign
                        new_d = d.copy()
                        new_d["label"] = w1
                        final_detections.append(new_d)
                        last_emitted_word = w1
                    elif last_emitted_word == w1:
                        # Handle bigram: Suppress w1, emit w2
                        new_d = d.copy()
                        new_d["label"] = w2
                        final_detections.append(new_d)
                        last_emitted_word = w2
                    else:
                        # Handle bigram: Emit both words
                        d1 = d.copy()
                        d1["label"] = w1
                        d2 = d.copy()
                        d2["label"] = w2
                        final_detections.extend([d1, d2])
                        last_emitted_word = w2
                else:
                    final_detections.append(d)
                    last_emitted_word = parts[-1]
            else:
                final_detections.append(d)
                last_emitted_word = label
                
        return final_detections


# ── Helpers ───────────────────────────────────────────────────────────────────

def _interval_overlap_ratio(s1, e1, s2, e2):
    """Returns Intersection over Minimum (IoM) for two intervals."""
    inter = max(0, min(e1, e2) - max(s1, s2))
    min_len = min(e1 - s1 + 1, e2 - s2 + 1)
    return inter / min_len if min_len > 0 else 0.0
