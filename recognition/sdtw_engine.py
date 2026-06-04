import numpy as np

class StreamingSDTW:
    """Incremental SDTW computation for a single template."""
    
    def __init__(self, template: np.ndarray, weights: np.ndarray, threshold: float, promising_ratio: float = 0.7):
        self.template = template            # (M, 56)
        self.M = len(template)
        self.weights = weights              # (56,)
        self.threshold = threshold
        self.promising_ratio = promising_ratio
        
        self.in_detection_zone = False
        self.patience_counter = 0
        self.PATIENCE_MAX = 5  # frames to wait after valley bottom
        self.reset()
    
    def reset(self):
        """Initialize/reset the DTW column."""
        self.prev_col = np.zeros(self.M + 1)
        self.prev_col[1:] = np.inf
        self.frames_fed = 0
        self.min_dist = float('inf')
        self.min_dist_frame = -1
        self.in_detection_zone = False
        self.patience_counter = 0
    
    def feed(self, frame: np.ndarray) -> dict:
        """Process one frame. Returns detection dict if match found, else None."""
        curr_col = np.zeros(self.M + 1)
        curr_col[0] = 0  # SDTW: can start matching at any point
        
        # Calculate costs: Weighted Euclidean for fingers, Angular for IMUs
        costs = np.zeros(self.M)
        
        # 1. Finger distances (weighted squared Euclidean)
        finger_idx = list(range(0, 16)) + list(range(28, 44))
        diff_fingers = frame[finger_idx] - self.template[:, finger_idx]
        costs += np.sum(self.weights[finger_idx] * (diff_fingers ** 2), axis=1)
        
        # 2. IMU distances (Angular: 1 - |dot product|)
        imu_starts = [16, 20, 24, 44, 48, 52]
        for s in imu_starts:
            imu_weight = np.sum(self.weights[s:s+4])
            if imu_weight > 0:
                dot = np.sum(frame[s:s+4] * self.template[:, s:s+4], axis=1)
                dist_quat = 1.0 - np.abs(dot)
                costs += imu_weight * dist_quat
        
        for j in range(1, self.M + 1):
            curr_col[j] = costs[j-1] + min(
                self.prev_col[j],        # insertion
                curr_col[j-1],           # deletion
                self.prev_col[j-1]       # diagonal match
            )
        
        self.prev_col = curr_col
        self.frames_fed += 1
        
        normalized_dist = curr_col[self.M] / self.M
        
        # Valley Detection Logic
        if normalized_dist < self.threshold and self.frames_fed >= self.M * 0.5:
            if not self.in_detection_zone:
                self.in_detection_zone = True
                self.patience_counter = 0
                
            if normalized_dist < self.min_dist:
                # We found a new bottom of the valley! Reset patience.
                self.min_dist = normalized_dist
                self.min_dist_frame = self.frames_fed
                self.patience_counter = 0
            else:
                # Distance is going up. 
                self.patience_counter += 1
                
            if self.patience_counter >= self.PATIENCE_MAX:
                # We've climbed out of the valley. Emit the absolute best match found.
                detection = {
                    'end_frame': self.min_dist_frame,
                    'start_frame_est': self.min_dist_frame - self.M,
                    'distance': self.min_dist,
                }
                self.reset()
                return detection
        else:
            if self.in_detection_zone:
                # We popped back out above threshold, valley is definitely over.
                detection = {
                    'end_frame': self.min_dist_frame,
                    'start_frame_est': self.min_dist_frame - self.M,
                    'distance': self.min_dist,
                }
                self.reset()
                return detection
                
            # If not in detection zone, just track min_dist for logging/promising checks
            if normalized_dist < self.min_dist:
                self.min_dist = normalized_dist
                self.min_dist_frame = self.frames_fed
                
        return None
        
    def force_emit(self) -> dict:
        """Force emission if currently in the detection zone, then reset."""
        if self.in_detection_zone:
            detection = {
                'end_frame': self.min_dist_frame,
                'start_frame_est': self.min_dist_frame - self.M,
                'distance': self.min_dist,
            }
            self.reset()
            return detection
            
        self.reset()
        return None
        
    @property
    def is_promising(self) -> bool:
        return self.min_dist < self.threshold * self.promising_ratio

class SDTWEngine:
    """Manages multiple StreamingSDTW instances for candidate templates."""
    
    def __init__(self, all_templates: dict, weights: np.ndarray, config):
        self.all_templates = all_templates  # {label: [template1, ...]}
        self.weights = weights
        self.threshold = config.SDTW_DETECTION_THRESHOLD
        self.promising_ratio = config.SDTW_PROMISING_RATIO
        self.active_matchers = {}           # {(label, idx): StreamingSDTW}
        self.frozen_matchers = {}           # {(label, idx): StreamingSDTW}
        self.frame_counter = 0
        
    def update_candidates(self, candidate_labels: list):
        """Called when pre-filter produces new top-K candidates."""
        new_keys = set()
        for label in candidate_labels:
            for idx, tmpl in enumerate(self.all_templates.get(label, [])):
                key = (label, idx)
                new_keys.add(key)
                if key not in self.active_matchers:
                    if key in self.frozen_matchers:
                        # Restore frozen state without reset
                        self.active_matchers[key] = self.frozen_matchers.pop(key)
                    else:
                        self.active_matchers[key] = StreamingSDTW(
                            tmpl, self.weights, self.threshold, self.promising_ratio
                        )
                    
        # Freeze matchers no longer in candidate set (unless promising)
        to_freeze = []
        for key in self.active_matchers:
            if key not in new_keys and not self.active_matchers[key].is_promising:
                to_freeze.append(key)
        for key in to_freeze:
            self.frozen_matchers[key] = self.active_matchers.pop(key)
            
    def feed(self, frame: np.ndarray, global_frame: int) -> list:
        """Feed one frame to all active matchers. Returns list of detections."""
        self.frame_counter = global_frame
        detections = []
        
        for (label, idx), matcher in list(self.active_matchers.items()):
            result = matcher.feed(frame)
            if result is not None:
                result['label'] = label
                result['template_idx'] = idx
                result['global_end_frame'] = self.frame_counter
                result['global_start_frame_est'] = self.frame_counter - matcher.M
                detections.append(result)
                
        return detections
        
    def reset_all(self):
        """Reset all matchers (e.g., when transitioning from IDLE to ACTIVE)."""
        for matcher in self.active_matchers.values():
            matcher.reset()
        for matcher in self.frozen_matchers.values():
            matcher.reset()
            
    def force_emit_all(self, global_frame: int) -> list:
        """Force all matchers in a valley to emit, then reset all matchers. Used when transitioning to IDLE."""
        self.frame_counter = global_frame
        detections = []
        
        for (label, idx), matcher in list(self.active_matchers.items()):
            result = matcher.force_emit()
            if result is not None:
                result['label'] = label
                result['template_idx'] = idx
                result['global_end_frame'] = self.frame_counter
                result['global_start_frame_est'] = self.frame_counter - matcher.M
                detections.append(result)
                
        for matcher in self.frozen_matchers.values():
            matcher.reset()
            
        return detections

def compute_static_dtw(test_frames: np.ndarray, template: np.ndarray, weights: np.ndarray) -> float:
    """
    Vectorized full DTW between a test segment and a template.
    Used by NMS to break ties between competing detections.
    Uses the same weighted Euclidean + angular-IMU cost as StreamingSDTW.
    """
    N, M = len(test_frames), len(template)
    if N == 0 or M == 0:
        return float('inf')
    
    finger_idx = list(range(0, 16)) + list(range(28, 44))
    imu_starts = [16, 20, 24, 44, 48, 52]

    # ── Precompute cost matrix fully vectorized (N x M) ──────────────────────
    # Finger cost: broadcast (N,F) - (M,F) → (N,M,F) → weighted sum → (N,M)
    diff = test_frames[:, np.newaxis, :][:, :, finger_idx] - template[np.newaxis, :, :][:, :, finger_idx]
    cost_matrix = np.sum(weights[finger_idx] * (diff ** 2), axis=2)  # (N, M)

    # IMU angular cost
    for s in imu_starts:
        imu_weight = np.sum(weights[s:s+4])
        if imu_weight > 0:
            # (N,4) dot (M,4) → (N,M)
            dot = test_frames[:, np.newaxis, s:s+4] * template[np.newaxis, :, s:s+4]
            dot = np.sum(dot, axis=2)
            cost_matrix += imu_weight * (1.0 - np.abs(dot))
    
    # ── DP path ──────────────────────────────────────────────────────────────
    dtw = np.full((N + 1, M + 1), np.inf)
    dtw[0, 0] = 0.0
    
    # Vectorize row-by-row (M loop stays, N is lifted to rows)
    for i in range(1, N + 1):
        prev_row = dtw[i - 1, :]        # (M+1,)
        curr_row = np.full(M + 1, np.inf)
        curr_row[0] = np.inf            # standard DTW (not SDTW) for tie-breaking
        for j in range(1, M + 1):
            curr_row[j] = cost_matrix[i-1, j-1] + min(
                prev_row[j],
                curr_row[j-1],
                prev_row[j-1]
            )
        dtw[i, :] = curr_row

    return dtw[N, M] / M
