import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pipeline_config import FEATURE_GROUPS, FEATURE_GROUP_ALIASES

def apply_feature_mask(frames: np.ndarray, disabled_groups: list) -> np.ndarray:
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
            print(f"Unknown feature group: {group}")
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
    was_1d = frames.ndim == 1
    out = np.atleast_2d(frames).copy().astype(float)

    mean = norm_stats['mean']   # (56,)
    std  = norm_stats['std']    # (56,)

    # Identify dead channels (std ~= 1e-6 means clamped due to zero variance)
    #Recheck the std should be that of the sensor data per sign not of all frames across all signs
    dead_mask = std < 0.01   # finger channels with no real variation

    out = (out - mean) / np.where(dead_mask, 1.0, std)

    out[:, dead_mask] = 0.0

    return out[0] if was_1d else out

def compute_normalization_stats(all_frames_list: list) -> dict:
    if not all_frames_list:
        return {'mean': np.zeros(56), 'std': np.ones(56)}

    combined = np.vstack(all_frames_list)

    finger_indices = list(range(0, 16)) + list(range(28, 44))
    imu_indices    = list(range(16, 28)) + list(range(44, 56))

    mean = np.zeros(56)
    std  = np.ones(56)

    mean[finger_indices] = np.mean(combined[:, finger_indices], axis=0)
    pre_std = np.std(combined[:, finger_indices], axis=0)
    # Use real std only where variation exists; dead channels get std=1.0
    # (normalize_frames will then zero them out via the dead_mask logic)
    #Recheck should leave std as it is and leave normalize frames to zero them out
    std[finger_indices] = np.where(pre_std > 1.0, pre_std, 1.0)

    # IMU channels: pass-through
    mean[imu_indices] = 0.0
    std[imu_indices]  = 1.0

    return {'mean': mean, 'std': std}

def resample_to_uniform(timestamps: np.ndarray, values: np.ndarray, target_hz: float = 50.0) -> tuple:
    #Interpolate irregularly-spaced frames to a uniform time grid.
    #Fingers use linear interpolation. Quaternions use SLERP.
    
    #Reckeck, are these timestamps per sign so the logic is correct to resample each sign independently? Yes, these are per sign, so the logic is correct. Each sign's frames are resampled independently, and the timestamps are relative to the start of that sign. The dt capping prevents long gaps within a sign from causing excessive interpolation points, which could happen if there are missing frames or lag during data collection.
    raw_dt = np.diff(timestamps) / 1e6
    raw_dt = np.where(raw_dt > 0.1, 0.02, raw_dt)
    
    t_sec = np.zeros(len(timestamps))
    if len(timestamps) > 1:
        t_sec[1:] = np.cumsum(raw_dt)
    
    # To ensure that t_sec is monotonic so no duplicate timestamps
    valid_idx = np.concatenate(([True], np.diff(t_sec) > 0))
    t_sec = t_sec[valid_idx]
    values = values[valid_idx]

    if len(t_sec) < 2:
        return t_sec, values

    dt = 1.0 / target_hz
    uniform_t = np.arange(0, t_sec[-1], dt)
    uniform_v = np.zeros((len(uniform_t), values.shape[1]))
    
    finger_channels = list(range(0, 16)) + list(range(28, 44))
    for ch in finger_channels:
        uniform_v[:, ch] = np.interp(uniform_t, t_sec, values[:, ch])
        
    imu_starts = [16, 20, 24, 44, 48, 52]
    for q_start in imu_starts:
        q_data_wxyz = values[:, q_start:q_start+4]
        if np.all(q_data_wxyz == 0.0):
            continue
            
        q_data_xyzw = np.column_stack((q_data_wxyz[:, 1:], q_data_wxyz[:, 0]))
        
        norms = np.linalg.norm(q_data_xyzw, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        q_data_xyzw /= norms
        
        rot = R.from_quat(q_data_xyzw)
        slerp = Slerp(t_sec, rot)
        
        interp_rot = slerp(uniform_t)
        interp_xyzw = interp_rot.as_quat()
        
        interp_wxyz = np.column_stack((interp_xyzw[:, 3], interp_xyzw[:, :3]))
        uniform_v[:, q_start:q_start+4] = interp_wxyz
        
    # Enforce qw >= 0
    for q_start in imu_starts:
        if not np.all(uniform_v[:, q_start:q_start+4] == 0.0):
            uniform_v[:, q_start:q_start+4] = normalize_quaternion_sign(
                uniform_v[:, q_start:q_start+4]
            )
        
    return uniform_t, uniform_v
    
def preprocess_stream(timestamps: np.ndarray, raw_frames: np.ndarray, disabled_groups: list = [], norm_stats: dict = None) -> tuple:
    masked = apply_feature_mask(raw_frames, disabled_groups)
    t, v = resample_to_uniform(timestamps, masked, target_hz=50.0)
    return t, v
