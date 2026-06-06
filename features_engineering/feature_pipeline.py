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


IMU_STARTS   = [16, 20, 24, 44, 48, 52]
HAND_IMU_STARTS = [16, 44]

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

    fingers = list(range(0, 16)) + list(range(28, 44))
    processed[:, fingers] -= frames[0, fingers]

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
        x=  np.zeros_like(values)
        return x, x
    velocity = np.gradient(values, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    return velocity, acceleration 


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


def isolate_gravity(frames: np.ndarray) -> np.ndarray:
    N = len(frames)
    processed = np.zeros((N, len(HAND_IMU_STARTS) * 3))
    world_g = np.array([0.0, -1.0, 0.0])
    for gi, qs in enumerate(HAND_IMU_STARTS):
        for t in range(N):
            rot = wxyz_to_xyzw(frames[t, qs:qs+4])
            processed[t, gi*3:(gi+1)*3] = rot.inv().apply(world_g)
    return processed


def log_transform_hall(frames: np.ndarray) -> np.ndarray:
    fingers = list(range(0, 16)) + list(range(28, 44))
    vals = frames[:, fingers].copy()
    return np.sign(vals) * np.log1p(np.abs(vals))


def _finger_group_magnitude(velocity_window, slc):
    return np.linalg.norm(velocity_window[:, slc], axis=1)


def compute_cross_correlations(finger_velocity: np.ndarray) -> np.ndarray:
    corrs = []
    right_fingers=[slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 16)]
    left_fingers=[slice(28, 31), slice(31, 34), slice(34, 37), slice(37, 40), slice(40, 44)]
    for groups in [right_fingers, left_fingers]:
        magnitudes = [_finger_group_magnitude(finger_velocity, g) for g in groups]
        for i, j in combinations(range(5), 2):
            magnitude1, magnitude2 = magnitudes[i], magnitudes[j]
            if np.std(magnitude1) < 1e-8 or np.std(magnitude2) < 1e-8:
                corrs.append(0.0)
            else:
                c = np.corrcoef(magnitude1, magnitude2)[0, 1]
                corrs.append(0.0 if np.isnan(c) else c)
    return np.array(corrs)


def compute_zero_crosing(velocity: np.ndarray) -> np.ndarray:
    if len(velocity) < 2:
        return np.zeros(velocity.shape[1])
    signs = np.sign(velocity)
    crossings = np.abs(np.diff(signs, axis=0)) > 0
    return np.sum(crossings, axis=0).astype(float) / (len(velocity) - 1)

def statistical_pool(ts: np.ndarray) -> np.ndarray:
    N, C = ts.shape
    processed = np.zeros(C * STATS_PER_CH)
    for c in range(C):
        col = ts[:, c]
        b = c * STATS_PER_CH
        processed[b+0] = np.mean(col)
        processed[b+1] = np.var(col)
        processed[b+2] = np.subtract(*np.percentile(col, [75, 25]))
        processed[b+3] = float(skew(col)) if N > 2 else 0.0
        processed[b+4] = float(kurtosis(col)) if N > 2 else 0.0
        processed[b+5] = np.sqrt(np.mean(col ** 2))   

        if N >= 4:
            fft_mag = np.abs(np.fft.rfft(col))[1:]
            top_k = min(N_FFT_COEFFS, len(fft_mag))
            top_idx = np.argsort(fft_mag)[-top_k:][::-1]
            processed[b+6:b+6+top_k] = fft_mag[top_idx]
    return processed


def extract_features(raw_frames: np.ndarray, dt: float = 0.02) -> np.ndarray | None:
    if len(raw_frames) < 5:
       return None

    fingers=list(range(0, 16)) + list(range(28, 44))
    delta = delta_reference(raw_frames)

    finger_vel, finger_acc = compute_derivatives(delta[:, fingers], dt)

    dir_vecs = quaternions_to_vectors(raw_frames)
    dir_vel, _ = compute_derivatives(dir_vecs, dt)

    gravity = isolate_gravity(raw_frames)

    log_hall = log_transform_hall(delta)

    full_vel, _ = compute_derivatives(raw_frames, dt)
    cross_corr = compute_cross_correlations(full_vel)

    zcr = compute_zero_crosing(finger_vel)

    pooled = np.concatenate([
        statistical_pool(delta[:, fingers]),   
        statistical_pool(dir_vecs),               
        statistical_pool(finger_vel),             
        statistical_pool(finger_acc),             
        statistical_pool(dir_vel),                
        statistical_pool(gravity),                
        statistical_pool(log_hall),               
    ])

    feature_vec = np.concatenate([pooled, cross_corr, zcr])

    return np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)


def feature_vector_length() -> int:
    n_pooled_ch = 32 + 36 + 32 + 32 + 36 + 6 + 32 
    n_cross_cooreltion     = 20
    n_zero_crossing       = 32                    
    return n_pooled_ch * STATS_PER_CH + n_cross_cooreltion + n_zero_crossing
