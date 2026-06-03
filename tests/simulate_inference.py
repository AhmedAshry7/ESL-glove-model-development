import os
import sys
import json
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.continuous_recognizer import ContinuousRecognizer
from preprocessing.stream_preprocessor import preprocess_stream
import config.pipeline_config as cfg

def main():
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    raw_dir   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

    print("Loading Continuous Recognizer...")
    try:
        recognizer = ContinuousRecognizer(model_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the prefilter and build the template database first!")
        return

    test_dir = os.path.join(raw_dir, "test")
    if not os.path.exists(test_dir):
        print(f"Test directory not found at {test_dir}")
        return

    json_files = [f for f in os.listdir(test_dir) if f.endswith('.json') and not f.endswith('_meta.json')]
    if not json_files:
        print(f"No .json files found in {test_dir}.")
        return

    json_files.sort()

    for filename in json_files:
        test_file = os.path.join(test_dir, filename)
        print(f"\n{'='*80}")
        print(f"Loading test file: {test_file}")
        print(f"{'='*80}")

        with open(test_file, 'r') as f:
            signs_data = json.load(f)

        for sign in signs_data:
            label = sign.get("label", "UNKNOWN")
            frames_data = np.array(sign.get("frames", []))
            if len(frames_data) == 0:
                continue

            print(f"\n{'='*60}")
            print(f"▶ Simulating detection for recorded sign: '{label}'")
            print(f"{'='*60}")

            timestamps = frames_data[:, 0]
            raw_frames = frames_data[:, 1:]

            # Preprocess WITHOUT normalization — the recognizer normalizes internally
            t, v = preprocess_stream(timestamps, raw_frames, disabled_groups=cfg.DISABLED_FEATURE_GROUPS)
            print(f"Preprocessed into {len(v)} frames (~{len(v)/cfg.TARGET_SAMPLE_HZ:.2f} seconds)")

            # ── Full state reset for this independent clip ──────────────────
            recognizer._reset_state()

            print("\nStarting simulation...")
            time.sleep(1)

            for i, frame in enumerate(v):
                # Simulate real-time delay (50Hz = 20ms)
                time.sleep(0.02)

                emissions = recognizer.feed_frame(frame)

                # Print status periodically or when an emission occurs
                if i % 10 == 0 or emissions:
                    print(f"\n--- [Frame {i}/{len(v)}] " + "-"*40)
                    print(f"Activity State: {'ACTIVE' if recognizer.was_active else 'IDLE'}")

                    fill_pct = recognizer.rolling_stats.fill_ratio * 100
                    gate_ok  = recognizer.rolling_stats.fill_ratio >= cfg.PREFILTER_MIN_FILL_RATIO

                    if recognizer.current_candidates and gate_ok:
                        top_str = ", ".join([
                            f"{c} ({p*100:.1f}%)"
                            for c, p in zip(recognizer.current_candidates[:5], recognizer.current_probs[:5])
                        ])
                        print(f"Prefilter Thinking: {top_str}")
                    elif not gate_ok:
                        print(f"Prefilter Thinking: Waiting for buffer ({fill_pct:.0f}% / {cfg.PREFILTER_MIN_FILL_RATIO*100:.0f}% required)")
                    else:
                        print(f"Prefilter Thinking: No candidates yet")

                    active_matchers = recognizer.sdtw_engine.active_matchers
                    if active_matchers:
                        dists = []
                        for key, matcher in active_matchers.items():
                            current_dist = matcher.prev_col[matcher.M] / matcher.M
                            min_dist = matcher.min_dist
                            dists.append((key[0], current_dist, min_dist))

                        dists.sort(key=lambda x: x[1])
                        print("SDTW Distances (Lower is better):")
                        for d_lbl, c_dist, m_dist in dists[:5]:
                            if c_dist < 1000:  # Ignore inf
                                print(f"   -> {d_lbl:<10} | Current: {c_dist:.3f} | Min Seen: {m_dist:.3f}")

                if emissions:
                    print("\n" + "⭐"*30)
                    for e in emissions:
                        conf = 1.0 - e['distance']
                        print(f"🌟 DECISION MADE: '{e['label']}'")
                        print(f"   Confidence: {conf*100:.1f}%  (DTW Distance: {e['distance']:.3f})")
                    print("⭐"*30 + "\n")

        print(f"\n--- Finished clip for '{label}' ---\n")
        time.sleep(1.5)

if __name__ == "__main__":
    main()

