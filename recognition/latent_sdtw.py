import numpy as np

class LatentStreamingSDTW:
    def __init__(self, template: np.ndarray, threshold: float, promising_ratio: float = 0.7):
        self.template = template            # (M, 128)
        self.M = len(template)
        self.threshold = threshold
        self.promising_ratio = promising_ratio
        
        self.in_detection_zone = False
        self.patience_counter = 0
        self.PATIENCE_MAX = 5
        self.reset()
        
        # Precompute template norms for faster cosine sim
        self.template_norms = np.linalg.norm(self.template, axis=1)
        self.template_norms[self.template_norms == 0] = 1e-10
    
    def reset(self):
        self.prev_col = np.zeros(self.M + 1)
        self.prev_col[1:] = np.inf
        self.frames_fed = 0
        self.min_dist = float('inf')
        self.min_dist_frame = -1
        self.in_detection_zone = False
        self.patience_counter = 0
    
    def feed(self, latent_frame: np.ndarray) -> dict:
        """Process one latent frame of shape (128,)."""
        curr_col = np.zeros(self.M + 1)
        curr_col[0] = 0
        
        frame_norm = np.linalg.norm(latent_frame)
        if frame_norm == 0:
            frame_norm = 1e-10
            
        # Cosine distance: 1 - (A.B)/(|A||B|)
        dots = np.sum(self.template * latent_frame, axis=1)
        cos_sim = dots / (self.template_norms * frame_norm)
        costs = 1.0 - cos_sim
        
        for j in range(1, self.M + 1):
            curr_col[j] = costs[j-1] + min(
                self.prev_col[j],        # insertion
                curr_col[j-1],           # deletion
                self.prev_col[j-1]       # diagonal match
            )
        
        self.prev_col = curr_col
        self.frames_fed += 1
        
        normalized_dist = curr_col[self.M] / self.M
        
        if normalized_dist < self.threshold and self.frames_fed >= self.M * 0.5:
            if not self.in_detection_zone:
                self.in_detection_zone = True
                self.patience_counter = 0
                
            if normalized_dist < self.min_dist:
                self.min_dist = normalized_dist
                self.min_dist_frame = self.frames_fed
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.PATIENCE_MAX:
                detection = {
                    'end_frame': self.min_dist_frame,
                    'start_frame_est': self.min_dist_frame - self.M,
                    'distance': self.min_dist,
                    'template_length': self.M,
                }
                self.reset()
                return detection
        else:
            if self.in_detection_zone:
                detection = {
                    'end_frame': self.min_dist_frame,
                    'start_frame_est': self.min_dist_frame - self.M,
                    'distance': self.min_dist,
                    'template_length': self.M,
                }
                self.reset()
                return detection
                
            if normalized_dist < self.min_dist:
                self.min_dist = normalized_dist
                self.min_dist_frame = self.frames_fed
                
        return None
        
    def force_emit(self) -> dict:
        if self.in_detection_zone:
            detection = {
                'end_frame': self.min_dist_frame,
                'start_frame_est': self.min_dist_frame - self.M,
                'distance': self.min_dist,
                'template_length': self.M,
            }
            self.reset()
            return detection
        self.reset()
        return None
        
    @property
    def is_promising(self) -> bool:
        return self.min_dist < self.threshold * self.promising_ratio

class LatentSDTWEngine:
    def __init__(self, templates: dict, default_threshold: float = 0.5):
        self.templates = templates
        self.default_threshold = default_threshold
        self.active_matchers = {}
        
        # Initialize all matchers immediately (since we aren't using a prefilter here)
        for label, tmpls in self.templates.items():
            for idx, tmpl in enumerate(tmpls):
                key = (label, idx)
                self.active_matchers[key] = LatentStreamingSDTW(tmpl, threshold=self.default_threshold)
                
        self.frame_counter = 0
        
    def feed(self, latent_frame: np.ndarray, global_frame: int) -> list:
        self.frame_counter = global_frame
        detections = []
        
        for (label, idx), matcher in self.active_matchers.items():
            result = matcher.feed(latent_frame)
            if result is not None:
                result['label'] = label
                result['template_idx'] = idx
                result['global_end_frame'] = self.frame_counter
                result['global_start_frame_est'] = self.frame_counter - matcher.M
                detections.append(result)
                
        return detections
