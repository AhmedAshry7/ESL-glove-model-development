"""
feature_pipeline.py

Transforms a variable-length window of raw 56-D glove frames into a
fixed-size feature vector suitable for a Random-Forest classifier.

Pipeline steps:
  1. Per-window delta referencing (zero-center to t=0)
  2. Derivatives (velocity, acceleration)
  3. Quaternion → forward / up direction vectors
  4. Gravity-vector isolation from hand IMUs
  5. Log-transformed Hall (finger) channels
  6. Inter-finger velocity cross-correlations
  7. Zero-crossing rates
  8. Statistical pooling → fixed-size vector
"""

import numpy as np
from scipy.stats import skew, kurtosis
from scipy.spatial.transform import Rotation
from itertools import combinations

FINGER_RIGHT = list(range(0, 16))
FINGER_LEFT  = list(range(28, 44))
FINGER_ALL   = FINGER_RIGHT + FINGER_LEFT

IMU_STARTS   = [16, 20, 24, 44, 48, 52]
HAND_IMU_STARTS = [16, 44]

RIGHT_FINGER_GROUPS = [slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 16)]
LEFT_FINGER_GROUPS = [slice(28, 31), slice(31, 34), slice(34, 37), slice(37, 40), slice(40, 44)]

N_FFT_COEFFS = 3
STATS_PER_CH = 6 + N_FFT_COEFFS

def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def wxyz_to_xyzw(q_wxyz):
    n = np.linalg.norm(q_wxyz)
    if n < 1e-8:
        return Rotation.identity()
    q = q_wxyz / n
    return Rotation.from_quat([q[1], q[2], q[3], q[0]])


def delta_reference(frames: np.ndarray) -> np.ndarray:
    processed = frames.copy()

    processed[:, FINGER_ALL] -= frames[0, FINGER_ALL]

    for quantules in IMU_STARTS:
        quantule = frames[0, quantules:quantules+4].copy()
        n = np.linalg.norm(quantule)
        if n < 1e-8:
            continue
        quantule /= n
        quantule_inverse = quat_conjugate(quantule)
        for t in range(len(processed)):
            qt = processed[t, quantules:quantules+4]
            qn = np.linalg.norm(qt)
            if qn < 1e-8:
                continue
            qt = qt / qn
            processed[t, quantules:quantules+4] = quat_multiply(quantule_inverse, qt)
    return processed

def compute_derivatives(values: np.ndarray, dt: float = 0.02):
    if len(values) < 3:
        z = np.zeros_like(values)
        return z, z
    vel = np.gradient(values, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    return vel, acc


def quaternions_to_vectors(frames: np.ndarray) -> np.ndarray:
    N = len(frames)
    processed = np.zeros((N, len(IMU_STARTS) * 6))
    for qi, qs in enumerate(IMU_STARTS):
        for t in range(N):
            rot = wxyz_to_xyzw(frames[t, qs:qs+4])
            fwd = rot.apply([0, 0, 1])
            up  = rot.apply([0, 1, 0])
            c = qi * 6
            processed[t, c:c+3] = fwd
            processed[t, c+3:c+6] = up
    return processed


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4a — Gravity-vector isolation
# ═══════════════════════════════════════════════════════════════════════════════

def isolate_gravity(frames: np.ndarray) -> np.ndarray:
    """Gravity direction in the hand IMU sensor frame (6 channels)."""
    N = len(frames)
    processed = np.zeros((N, len(HAND_IMU_STARTS) * 3))
    world_g = np.array([0.0, -1.0, 0.0])
    for gi, qs in enumerate(HAND_IMU_STARTS):
        for t in range(N):
            rot = wxyz_to_xyzw(frames[t, qs:qs+4])
            processed[t, gi*3:(gi+1)*3] = rot.inv().apply(world_g)
    return processed


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4b — Log-transformed Hall sensors
# ═══════════════════════════════════════════════════════════════════════════════

def log_transform_hall(frames: np.ndarray) -> np.ndarray:
    """log(1+|x|)·sign(x) on delta-referenced finger channels."""
    vals = frames[:, FINGER_ALL].copy()
    return np.sign(vals) * np.log1p(np.abs(vals))


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4c — Inter-finger velocity cross-correlations
# ═══════════════════════════════════════════════════════════════════════════════

def _finger_group_magnitude(vel_window, slc):
    """RMS magnitude of a finger group's velocity across a window."""
    return np.linalg.norm(vel_window[:, slc], axis=1)


def compute_cross_correlations(finger_velocity: np.ndarray) -> np.ndarray:
    """Pearson correlation between every pair of fingers on each hand (20 values)."""
    corrs = []
    for groups in [RIGHT_FINGER_GROUPS, LEFT_FINGER_GROUPS]:
        mags = [_finger_group_magnitude(finger_velocity, g) for g in groups]
        for i, j in combinations(range(5), 2):
            m1, m2 = mags[i], mags[j]
            if np.std(m1) < 1e-8 or np.std(m2) < 1e-8:
                corrs.append(0.0)
            else:
                c = np.corrcoef(m1, m2)[0, 1]
                corrs.append(0.0 if np.isnan(c) else c)
    return np.array(corrs)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 4d — Zero-crossing rate
# ═══════════════════════════════════════════════════════════════════════════════

def compute_zcr(velocity: np.ndarray) -> np.ndarray:
    """ZCR per channel over the window (returns 1-D array of length C)."""
    if len(velocity) < 2:
        return np.zeros(velocity.shape[1])
    signs = np.sign(velocity)
    crossings = np.abs(np.diff(signs, axis=0)) > 0
    return np.sum(crossings, axis=0).astype(float) / (len(velocity) - 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Statistical pooling
# ═══════════════════════════════════════════════════════════════════════════════

def statistical_pool(ts: np.ndarray) -> np.ndarray:
    """Pool (N, C) time-series → (C * STATS_PER_CH,) vector."""
    N, C = ts.shape
    processed = np.zeros(C * STATS_PER_CH)
    for c in range(C):
        col = ts[:, c]
        b = c * STATS_PER_CH
        processed[b+0] = np.mean(col)
        processed[b+1] = np.var(col)
        processed[b+2] = np.subtract(*np.percentile(col, [75, 25]))  # IQR
        processed[b+3] = float(skew(col)) if N > 2 else 0.0
        processed[b+4] = float(kurtosis(col)) if N > 2 else 0.0
        processed[b+5] = np.sqrt(np.mean(col ** 2))                  # RMS
        # Dominant FFT coefficients (skip DC)
        if N >= 4:
            fft_mag = np.abs(np.fft.rfft(col))[1:]
            top_k = min(N_FFT_COEFFS, len(fft_mag))
            top_idx = np.argsort(fft_mag)[-top_k:][::-1]
            processed[b+6:b+6+top_k] = fft_mag[top_idx]
    return processed


# ═══════════════════════════════════════════════════════════════════════════════
#  Master function
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(raw_frames: np.ndarray, dt: float = 0.02) -> np.ndarray | None:
    """Full pipeline: (N, 56) raw frames → fixed-size feature vector.

    Returns *None* if the window is too short (< 5 frames).
    """
    if len(raw_frames) < 5:
        return None

    # 1. Delta reference
    delta = delta_reference(raw_frames)

    # 2. Derivatives of finger channels
    finger_vel, finger_acc = compute_derivatives(delta[:, FINGER_ALL], dt)

    # 3. Direction vectors & their velocity
    dir_vecs = quaternions_to_vectors(raw_frames)
    dir_vel, _ = compute_derivatives(dir_vecs, dt)

    # 4a. Gravity isolation
    gravity = isolate_gravity(raw_frames)

    # 4b. Log-transformed Hall sensors
    log_hall = log_transform_hall(delta)

    # 4c. Cross-correlations (on raw-frame finger velocity, not delta)
    full_vel, _ = compute_derivatives(raw_frames, dt)
    cross_corr = compute_cross_correlations(full_vel)

    # 4d. Zero-crossing rate on finger velocity
    zcr = compute_zcr(finger_vel)

    # 5. Statistical pooling of every time-series block
    pooled = np.concatenate([
        statistical_pool(delta[:, FINGER_ALL]),   # 32 ch
        statistical_pool(dir_vecs),               # 36 ch
        statistical_pool(finger_vel),             # 32 ch
        statistical_pool(finger_acc),             # 32 ch
        statistical_pool(dir_vel),                # 36 ch
        statistical_pool(gravity),                # 6 ch
        statistical_pool(log_hall),               # 32 ch
    ])

    # 6. Concatenate pooled + scalar features
    feature_vec = np.concatenate([pooled, cross_corr, zcr])

    return np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)


def feature_vector_length() -> int:
    """Return the expected length of the feature vector (for pre-allocation)."""
    n_pooled_ch = 32 + 36 + 32 + 32 + 36 + 6 + 32      # 206
    n_cross     = 20
    n_zcr       = 32                                      # finger velocity channels
    return n_pooled_ch * STATS_PER_CH + n_cross + n_zcr
