import argparse
import json
import pickle
import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

FLEX_KEYS   = ["flex8", "flex9", "flex10", "flex11",
               "flex12", "flex13", "flex14", "flex15"]
N_PADS      = 7
N_FEATURES  = 22          # 8 flex + 7 pad-z + 7 pad-r

SMOOTH_KERNEL = 3         # rolling-mean window for flex smoothing (must match training)



def parse_frame(frame):
    """
    Convert one raw frame dict to a float32 vector of length 22.
    """
    vec = np.zeros(N_FEATURES, dtype=np.float32)

    flex = frame["flex"]
    for i, key in enumerate(FLEX_KEYS):
        vec[i] = float(flex[key]["curl"])

    pad_lookup = {p["n"]: p for p in frame["pads"]}
    for j in range(N_PADS):
        pad = pad_lookup.get(f"PAD_CH{j}", {"z": -1, "r": 0})
        vec[8 + j]  = float(pad["z"])  
        vec[15 + j] = float(pad["r"]) 

    return vec


def parse_frames(frames):
    """Parse a list of frame dicts to a (T, 22) float32 array."""
    return np.stack([parse_frame(f) for f in frames], axis=0)



def normalize_sequence(seq):
    """
    Apply the same three normalization steps used during training:
    """
    out = seq.copy()

    # 1. Flex smoothing — rolling mean, window=SMOOTH_KERNEL
    #    Edge frames use available neighbours (no zero-padding at boundaries)
    T    = seq.shape[0]
    half = SMOOTH_KERNEL // 2
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        out[t, :8] = seq[lo:hi, :8].mean(axis=0)

    # 2. Pad-z normalization: raw z ∈ {-1, 0, 1, 2, 3, 4} → [0.0, 1.0]
    out[:, 8:15] = (seq[:, 8:15] + 1) / 5

    # 3. Pad-r normalization 
    r_part = out[:, 15:22]
    binned_r = np.zeros_like(r_part)
    
    # Binning logic
    binned_r[(r_part > 0) & (r_part < 900)] = 1  
    binned_r[r_part >= 900] = 2                 
    
    out[:, 15:22] = binned_r / 2.0

    # Clip to [0, 1] to guard against tiny floating-point overflows
    out = np.clip(out, 0.0, 1.0)
    return out


def extract_window(buffer, W):
    """
    Extract a (W, 22) window from the circular buffer.
    """
    T = buffer.shape[0]
    window = np.zeros((W, N_FEATURES), dtype=np.float32)
    mask   = np.zeros(W, dtype=bool)

    if T >= W:
        window[:] = buffer[-W:]     # take the most recent W frames
        mask[:]   = True
    else:
        window[W - T:] = buffer     # right-align real frames, zeros on left
        mask[W - T:]   = True

    return window[np.newaxis, :, :], mask[np.newaxis, :]   # add batch dim

class Debouncer:

    def __init__(self, n_classes, smoothing_window, confidence_threshold, cooldown_n):
        self.n_classes            = n_classes
        self.confidence_threshold = confidence_threshold
        self.cooldown_n           = cooldown_n
        self.prob_queue           = deque(maxlen=smoothing_window)
        self.cooldown_counter     = 0
        self.last_emitted         = None

    def push(self, probs):
        """
        Feed one window's softmax probability vector.
        Returns (word, confidence) if a word is emitted this step, or (None, 0.0) if suppressed or gated out.
        """
        max_prob = float(probs.max())

        # Stage 1: confidence gate — zero-vector signals "no sign" to the queue
        if max_prob < self.confidence_threshold:
            self.prob_queue.append(np.zeros(self.n_classes, dtype=np.float32))
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            return None, 0.0

        self.prob_queue.append(probs.astype(np.float32))

        # Stage 2: temporal smoothing — need a full queue before emitting
        if len(self.prob_queue) < self.prob_queue.maxlen:
            return None, 0.0

        avg_probs    = np.mean(self.prob_queue, axis=0)   # (n_classes,)
        final_idx    = int(np.argmax(avg_probs))
        final_prob   = float(avg_probs[final_idx])

        if final_prob < self.confidence_threshold:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
            return None, 0.0

        return self._apply_cooldown(final_idx, final_prob)

    def _apply_cooldown(self, class_idx, confidence):
        """Stage 3: suppress repeated emissions of the same word."""
        # Build a placeholder word token; caller maps idx→label
        if self.cooldown_counter > 0 and class_idx == self.last_emitted:
            self.cooldown_counter -= 1
            return None, 0.0   # suppressed — same word, in cooldown

        # Emit
        self.last_emitted     = class_idx
        self.cooldown_counter = self.cooldown_n
        return class_idx, confidence



class InferenceEngine:

    def __init__(self, model_path, processed_dir, confidence_threshold = 0.75, smoothing_window = 4, verbose = False):

        self.verbose    = verbose
        processed_dir   = Path(processed_dir)

        meta_path = processed_dir / "extracted_features_deep_learning_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.W           = meta["W"]
        self.N_CLASSES   = meta["n_classes"]
        self.CLASS_NAMES = meta["class_labels"]
        # Stride = 25% of W, same as training sliding-window assumption
        self.STRIDE      = max(1, self.W // 4)
        # Cooldown = W / STRIDE windows = 4  (one full sign-worth of windows)
        self.COOLDOWN_N  = self.W // self.STRIDE

        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
            print(f"  Architecture : {self.model.name}")
            print(f"  Input shape  : {self.model.input_shape}")
            print(f"  Output shape : {self.model.output_shape}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model from '{model_path}': {e}")

        self.debouncer = Debouncer(
            n_classes            = self.N_CLASSES,
            smoothing_window     = smoothing_window,
            confidence_threshold = confidence_threshold,
            cooldown_n           = self.COOLDOWN_N,
        )

        print(f"✓ Artifacts loaded from {processed_dir}")
        print(f"  W={self.W}  stride={self.STRIDE}  cooldown={self.COOLDOWN_N}")
        print(f"  Confidence threshold : {confidence_threshold}")
        print(f"  Smoothing window     : {smoothing_window} windows")
        print(f"  Classes ({self.N_CLASSES})       : {self.CLASS_NAMES}")


    def predict_stream(self, frames):
        raw_seq = parse_frames(frames)       # (T, 22) — raw units
        T       = raw_seq.shape[0]
        print(f"\nProcessing {T} frames …")

        norm_seq = normalize_sequence(raw_seq)  # (T, 22)

        emissions    = []
        window_index = 0
        for start in range(0, T, self.STRIDE):
            end          = start + self.W
            buffer_slice = norm_seq[start : end]  # up to W frames

            # Build padded window + mask
            window, mask = extract_window(buffer_slice, self.W)
            # window: (1, W, 22)  mask: (1, W)

            probs = self._forward(window)   # (N_CLASSES,)

            result_idx, confidence = self.debouncer.push(probs)

            if self.verbose:
                top_cls   = int(np.argmax(probs))
                top_prob  = float(probs[top_cls])
                top_word  = self.CLASS_NAMES[top_cls]
                emitted   = f"→ EMIT '{self.CLASS_NAMES[result_idx]}'" if result_idx is not None else "   (suppressed)"
                print(f"  window={window_index:4d}  frame={end-1:5d}  "
                      f"top={top_word}({top_prob:.2f})  {emitted}")

            if result_idx is not None:
                word = self.CLASS_NAMES[result_idx]
                emission = {
                    "word"        : word,
                    "confidence"  : round(confidence, 4),
                    "frame_index" : min(end - 1, T - 1),
                    "window_index": window_index,
                }
                emissions.append(emission)
                if not self.verbose:
                    print(f"  [{window_index:4d} | frame {end-1:5d}]  '{word}'  "
                          f"conf={confidence:.2%}")

            window_index += 1

        return emissions

    def _forward(self, window):
        """
        Run a single forward pass and return a (N_CLASSES,) float32 softmax vector.
        """
        probs = self.model(window, training=False).numpy()[0]   # (N_CLASSES,)
        return probs.astype(np.float32)


def print_summary(emissions):
    """Print a formatted summary table of all emissions."""
    if not emissions:
        print("\n⚠  No signs detected. Try lowering --confidence.")
        return

    sentence = " ".join(e["word"] for e in emissions)
    print("\n" + "═" * 62)
    print(f"  Detected sentence : {sentence}")
    print("═" * 62)
    print(f"  {'#':<4}  {'Word':<14}  {'Confidence':>11}  "
          f"{'Frame':>7}  {'Window':>7}")
    print("─" * 62)
    for i, e in enumerate(emissions, 1):
        print(f"  {i:<4}  {e['word']:<14}  {e['confidence']:>10.2%}  "
              f"{e['frame_index']:>7d}  {e['window_index']:>7d}")
    print("═" * 62)
    print(f"  Total signs emitted : {len(emissions)}")

frames_path = "data/inference_data.json"
if not frames_path.exists():
    print(f"ERROR: frames file not found: {frames_path}", file=sys.stderr)
    sys.exit(1)

with open(frames_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, list):
    frames = raw
elif isinstance(raw, dict) and "frames" in raw:
    frames = raw["frames"]
else:
    print("ERROR: JSON must be a list of frames or a dict with a 'frames' key.",
            file=sys.stderr)
    sys.exit(1)

if len(frames) == 0:
    print("ERROR: No frames found in the JSON file.", file=sys.stderr)
    sys.exit(1)

print(f"✓ Loaded {len(frames)} frames from {frames_path}")

engine = InferenceEngine(model_path = "models/best_BiLSTM.keras", processed_dir = "data/processed", confidence_threshold = 0.75, smoothing_window = 3, verbose = False)

emissions = engine.predict_stream(frames)

print_summary(emissions, engine.CLASS_NAMES)
