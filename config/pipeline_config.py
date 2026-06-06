
import numpy as np
import math

N_FINGER_SENSORS = 32        # 16 per hand
N_IMU_CHANNELS = 24          # 6 IMUs × 4 quaternion components (w, x, y, z)
N_FEATURES_PER_FRAME = 56

FEATURE_GROUPS = {
    'right_pinky':         slice(0, 3),
    'right_ring':          slice(3, 6),
    'right_middle':        slice(6, 9),
    'right_index':         slice(9, 12),
    'right_thumb':         slice(12, 16),
    'right_hand_imu':      slice(16, 20),
    'right_forearm_imu':   slice(20, 24),
    'right_upper_arm_imu': slice(24, 28),
    
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

TARGET_SAMPLE_HZ = 50.0

CHANNEL_WEIGHTS = np.zeros(N_FEATURES_PER_FRAME)
for side in ['right', 'left']:
    for f in ['pinky', 'ring', 'middle', 'index', 'thumb']:
        CHANNEL_WEIGHTS[FEATURE_GROUPS[f"{side}_{f}"]] = 1.0
    CHANNEL_WEIGHTS[FEATURE_GROUPS[f"{side}_hand_imu"]] = 0.8
    CHANNEL_WEIGHTS[FEATURE_GROUPS[f"{side}_forearm_imu"]] = 0.5
    CHANNEL_WEIGHTS[FEATURE_GROUPS[f"{side}_upper_arm_imu"]] = 0.3

for group in DISABLED_FEATURE_GROUPS:
    if group in FEATURE_GROUP_ALIASES:
        for sub in FEATURE_GROUP_ALIASES[group]:
            CHANNEL_WEIGHTS[FEATURE_GROUPS[sub]] = 0.0
    elif group in FEATURE_GROUPS:
        CHANNEL_WEIGHTS[FEATURE_GROUPS[group]] = 0.0

if np.sum(CHANNEL_WEIGHTS) > 0:
    CHANNEL_WEIGHTS /= np.sum(CHANNEL_WEIGHTS)

CONFIDENCE_THRESHOLD = 0.46
