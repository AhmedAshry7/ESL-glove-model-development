import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pipeline_config import FEATURE_GROUPS, FEATURE_GROUP_ALIASES

def apply_feature_mask(frames: np.ndarray, disabled_groups: list) -> np.ndarray:
    """Zero out channels belonging to disabled groups."""
    if not disabled_groups:
        return frames
    
    masked = frames.copy()
    for group in disabled_groups:
        if group in FEATURE_GROUP_ALIASES:
            for sub in FEATURE_GROUP_ALIASES[group]:
                masked[..., FEATURE_GROUPS[sub]] = 0.0
        elif group in FEATURE_GROUPS:
            masked[..., FEATURE_GROUPS[group]] = 0.0
        else:
            raise ValueError(f"Unknown feature group: {group}")
    return masked

def normalize_quaternion_sign(quats: np.ndarray) -> np.ndarray:
    """
    Enforce qw >= 0 convention to remove double-cover ambiguity.
    quats shape: (N, 4) where order is [qw, qx, qy, qz]
    """
    out = quats.copy()
    neg_w_idx = out[:, 0] < 0
    out[neg_w_idx] *= -1.0
    return out

def normalize_frames(frames: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    Apply z-score normalization to finger channels only.

    Quaternion (IMU) channels are left untouched — their angular-distance
    metric in the SDTW engine is already scale-invariant, and zeroing the
    mean of a unit-quaternion component would break that metric.

    Args:
        frames:     (N, 56) array of preprocessed frames
        norm_stats: dict with keys 'mean' and 'std', each shape (56,).
                    std values for IMU channels should be 1.0 so they pass
                    through unchanged.

    Returns:
        (N, 56) normalized array
    """
    out = frames.copy()
    mean = norm_stats['mean']   # (56,)
    std  = norm_stats['std']    # (56,) — IMU channels have std=1.0

    # Only apply to finger channels; IMU channels divide by 1.0 (no-op)
    out = (out - mean) / std
    return out

def compute_normalization_stats(all_frames_list: list) -> dict:
    """
    Compute per-channel mean and std from a list of frame arrays.
    Only finger channels (indices 0-15 and 28-43) get real statistics;
    IMU channels (16-27 and 44-55) get mean=0.0 and std=1.0 so that
    normalize_frames() is a no-op for them.

    Args:
        all_frames_list: list of (N_i, 56) arrays from all training recordings

    Returns:
        dict with 'mean' (56,) and 'std' (56,) arrays
    """
    if not all_frames_list:
        return {'mean': np.zeros(56), 'std': np.ones(56)}

    combined = np.vstack(all_frames_list)   # (total_frames, 56)

    finger_indices = list(range(0, 16)) + list(range(28, 44))
    imu_indices    = list(range(16, 28)) + list(range(44, 56))

    mean = np.zeros(56)
    std  = np.ones(56)   # default 1.0 → no-op for IMU channels

    mean[finger_indices] = np.mean(combined[:, finger_indices], axis=0)
    raw_std = np.std(combined[:, finger_indices], axis=0)
    # Clamp to avoid divide-by-zero on channels with no variation
    std[finger_indices] = np.where(raw_std > 1e-6, raw_std, 1e-6)

    # IMU channels: pass-through
    mean[imu_indices] = 0.0
    std[imu_indices]  = 1.0

    return {'mean': mean, 'std': std}

def resample_to_uniform(timestamps: np.ndarray, values: np.ndarray, target_hz: float = 50.0) -> tuple:
    """
    Interpolate irregularly-spaced frames to a uniform time grid.
    Fingers use linear interpolation. Quaternions use SLERP.
    
    Args:
        timestamps: (N,) ESP32 microsecond timestamps
        values: (N, 28) sensor values
        target_hz: target uniform sample rate
        
    Returns:
        (uniform_timestamps, uniform_values)
    """
    # Calculate dt and cap anomalous gaps (e.g., > 100ms) to 20ms 
    # to prevent interpolation from inflating the sequence length during UDP lag/disconnects
    raw_dt = np.diff(timestamps) / 1e6
    raw_dt = np.where(raw_dt > 0.1, 0.02, raw_dt)
    
    t_sec = np.zeros(len(timestamps))
    if len(timestamps) > 1:
        t_sec[1:] = np.cumsum(raw_dt)
    
    # Ensure t_sec is strictly increasing (filters out duplicate UDP packets)
    valid_idx = np.concatenate(([True], np.diff(t_sec) > 0))
    t_sec = t_sec[valid_idx]
    values = values[valid_idx]

    if len(t_sec) < 2:
        return t_sec, values

    dt = 1.0 / target_hz
    uniform_t = np.arange(0, t_sec[-1], dt)
    uniform_values = np.zeros((len(uniform_t), values.shape[1]))
    
    # Linear interpolation for all 32 finger channels (indices 0-15, and 28-43)
    finger_channels = list(range(0, 16)) + list(range(28, 44))
    for ch in finger_channels:
        uniform_values[:, ch] = np.interp(uniform_t, t_sec, values[:, ch])
        
    # SLERP for the 6 IMU quaternions (Right: 16, 20, 24 | Left: 44, 48, 52)
    # Our data order is [w, x, y, z]. Scipy expects [x, y, z, w].
    imu_starts = [16, 20, 24, 44, 48, 52]
    for q_start in imu_starts:
        q_data_wxyz = values[:, q_start:q_start+4]
        # Skip if all zeros (e.g. from feature masking)
        if np.all(q_data_wxyz == 0.0):
            continue
            
        q_data_xyzw = np.column_stack((q_data_wxyz[:, 1:], q_data_wxyz[:, 0]))
        
        # Slerp requires unit quaternions
        norms = np.linalg.norm(q_data_xyzw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q_data_xyzw /= norms
        
        rot = R.from_quat(q_data_xyzw)
        slerp = Slerp(t_sec, rot)
        
        interp_rot = slerp(uniform_t)
        interp_xyzw = interp_rot.as_quat()
        
        # Convert back to [w,x,y,z]
        interp_wxyz = np.column_stack((interp_xyzw[:, 3], interp_xyzw[:, :3]))
        uniform_values[:, q_start:q_start+4] = interp_wxyz
        
    # Enforce qw >= 0
    for q_start in imu_starts:
        if not np.all(uniform_values[:, q_start:q_start+4] == 0.0):
            uniform_values[:, q_start:q_start+4] = normalize_quaternion_sign(
                uniform_values[:, q_start:q_start+4]
            )
        
    return uniform_t, uniform_values
    
def preprocess_stream(timestamps: np.ndarray, raw_frames: np.ndarray, disabled_groups: list = [], norm_stats: dict = None) -> tuple:
    """Full pipeline: feature masking -> resample -> (optional) normalize."""
    masked = apply_feature_mask(raw_frames, disabled_groups)
    t, v = resample_to_uniform(timestamps, masked, target_hz=50.0)
    if norm_stats is not None and len(v) > 0:
        v = normalize_frames(v, norm_stats)
    return t, v
