"""
pipeline_config.py

Global configuration for the ESL-Glove SDTW Architecture v4.
Updated for Dual-Arm 56-dimensional telemetry.
"""
import numpy as np
import math

# ── Sensor Layout (56 Dimensions per frame) ──
N_FINGER_SENSORS = 32        # 16 per hand
N_IMU_CHANNELS = 24          # 6 IMUs × 4 quaternion components (w, x, y, z)
N_FEATURES_PER_FRAME = 56

# ── Feature Groups & Masking ──
FEATURE_GROUPS = {
    # Right Hand (Master)
    'right_pinky':         slice(0, 3),
    'right_ring':          slice(3, 6),
    'right_middle':        slice(6, 9),
    'right_index':         slice(9, 12),
    'right_thumb':         slice(12, 16),
    'right_hand_imu':      slice(16, 20),
    'right_forearm_imu':   slice(20, 24),
    'right_upper_arm_imu': slice(24, 28),
    
    # Left Hand (Slave)
    'left_pinky':         slice(28, 31),
    'left_ring':          slice(31, 34),
    'left_middle':        slice(34, 37),
    'left_index':         slice(37, 40),
    'left_thumb':         slice(40, 44),
    'left_hand_imu':      slice(44, 48),
    'left_forearm_imu':   slice(48, 52),
    'left_upper_arm_imu': slice(52, 56),
}

FEATURE_GROUP_ALIASES = {
    'all_right_fingers': ['right_pinky', 'right_ring', 'right_middle', 'right_index', 'right_thumb'],
    'all_right_imus':    ['right_hand_imu', 'right_forearm_imu', 'right_upper_arm_imu'],
    'all_left_fingers':  ['left_pinky', 'left_ring', 'left_middle', 'left_index', 'left_thumb'],
    'all_left_imus':     ['left_hand_imu', 'left_forearm_imu', 'left_upper_arm_imu'],
    'all_fingers':       ['all_right_fingers', 'all_left_fingers'],
    'all_imus':          ['all_right_imus', 'all_left_imus'],
}

DISABLED_FEATURE_GROUPS = []

# ── Sampling ──
TARGET_SAMPLE_HZ = 50.0

# ── Activity Monitor (Idle Detection) ──
# Raw-scale threshold — used ONLY in the simulation for test clips that are
# fed without normalization (e.g. the activity check in the old simulate_inference
# path). The recognizer feeds normalized frames, so it uses FINGER_IDLE_THRESHOLD_NORM.
FINGER_IDLE_THRESHOLD = 0.02
ARM_IDLE_THRESHOLD = 0.01
IDLE_FRAMES_REQUIRED = 15    # 15 frames @ 50Hz = 300ms

# Normalized-scale threshold — after z-score normalization the finger channels
# are in units of standard deviations. A frame-to-frame change of 0.05 sigma
# across active channels is a reasonable "still" boundary.
# Empirically measured: signing sequences have finger_vel ~0.06 at mid-sign.
FINGER_IDLE_THRESHOLD_NORM = 0.05

# ── Pre-Filter (Random Forest) ──
ROLLING_WINDOW_SIZE = 50       # 50 frames @ 50Hz = 1.0 second
PREFILTER_INTERVAL = 10        # Re-evaluate top-K candidates every 10 frames (200ms)

# Gate: don't emit predictions until the rolling buffer is at least this full.
# Prevents noisy early-window statistics from poisoning the prefilter.
PREFILTER_MIN_FILL_RATIO = 0.6   # 30 of 50 frames must be present

# Dynamic top-K: use 15% of vocabulary size, with a floor.
# With ≤30 classes, just activate all templates since SDTW is fast enough.
PREFILTER_TOP_K_RATIO = 0.15
PREFILTER_TOP_K_MIN   = 25

def get_top_k(n_classes: int) -> int:
    """Return the number of SDTW candidates to activate given vocabulary size."""
    return max(PREFILTER_TOP_K_MIN, math.ceil(PREFILTER_TOP_K_RATIO * n_classes))

# ── SDTW Engine ──
# After z-score normalization, the weighted Euclidean squared distance per frame
# is bounded by the channel weights. Empirically, correct-class templates score
# between 0.01 and 0.77 on test sequences, so we use 0.9 as the detection
# threshold with some headroom. Tune upward if you get too many false positives.
SDTW_DETECTION_THRESHOLD = 0.9   # Normalized DTW distance to trigger a match
SDTW_PROMISING_RATIO = 0.7       # Keep template active if min_dist < threshold * ratio

# Per-channel DTW weights (to be tuned via grid search).
CHANNEL_WEIGHTS = np.zeros(N_FEATURES_PER_FRAME)
for side in ['right', 'left']:
    for f in ['pinky', 'ring', 'middle', 'index', 'thumb']:
        CHANNEL_WEIGHTS[FEATURE_GROUPS[f"{side}_{f}"]] = 1.0
    CHANNEL_WEIGHTS[FEATURE_GROUPS[f"{side}_hand_imu"]] = 0.8
    CHANNEL_WEIGHTS[FEATURE_GROUPS[f"{side}_forearm_imu"]] = 0.5
    CHANNEL_WEIGHTS[FEATURE_GROUPS[f"{side}_upper_arm_imu"]] = 0.3

# Mask out weights for disabled groups
for group in DISABLED_FEATURE_GROUPS:
    if group in FEATURE_GROUP_ALIASES:
        for sub in FEATURE_GROUP_ALIASES[group]:
            CHANNEL_WEIGHTS[FEATURE_GROUPS[sub]] = 0.0
    elif group in FEATURE_GROUPS:
        CHANNEL_WEIGHTS[FEATURE_GROUPS[group]] = 0.0

# Normalize weights so they sum to 1.0, making the threshold scale-invariant
if np.sum(CHANNEL_WEIGHTS) > 0:
    CHANNEL_WEIGHTS /= np.sum(CHANNEL_WEIGHTS)

# ── Non-Maximum Suppression (NMS) ──
NMS_OVERLAP_THRESHOLD = 0.5       # Discard detection if IoU overlap > 0.5
NMS_COOLDOWN_FRAMES = 25          # 25 frames @ 50Hz = 0.5s cooldown between emissions
NMS_WINDOW_SECONDS = 3.0          # How far back to look for overlapping detections

# ── Post-Processing ──
CONFIDENCE_THRESHOLD = 0.46       # Minimum confidence to emit to UI/NLP
