import numpy as np

class ActivityMonitor:
    """Lightweight idle/active detector based on velocity thresholds."""
    
    def __init__(self, finger_thresh=0.03, arm_thresh=0.02, idle_frames_req=15, active_finger_channels=None, active_imu_channels=None):
        self.finger_thresh = finger_thresh
        self.arm_thresh = arm_thresh
        self.idle_frames_req = idle_frames_req
        
        # Default to all channels if none specified
        if active_finger_channels is None:
            self.active_finger_channels = list(range(0, 16)) + list(range(28, 44))
        else:
            self.active_finger_channels = active_finger_channels
            
        if active_imu_channels is None:
            imu_starts = [16, 20, 24, 44, 48, 52]
            self.active_imu_channels = []
            for s in imu_starts:
                self.active_imu_channels.extend(range(s, s+4))
        else:
            self.active_imu_channels = active_imu_channels
            
        self.idle_counter = 0
        self.is_active = False
        self.prev_frame = None
        
    def feed(self, frame: np.ndarray) -> bool:
        """
        Returns True if signer is currently active.
        frame: (56,) shaped array
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            return False
            
        delta = np.abs(frame - self.prev_frame)
        
        # Fingers: sum of absolute differences across active hands
        if self.active_finger_channels:
            finger_vel = np.sum(delta[self.active_finger_channels])
        else:
            finger_vel = 0.0
        
        # Arm IMUs: L2 norm of quaternion component differences across active arms
        if self.active_imu_channels:
            arm_vel = np.linalg.norm(delta[self.active_imu_channels])
        else:
            arm_vel = 0.0
        
        self.prev_frame = frame
        
        # Check thresholds (only if the threshold is > 0 and we actually have channels)
        finger_active = (len(self.active_finger_channels) > 0) and (finger_vel > self.finger_thresh)
        arm_active = (len(self.active_imu_channels) > 0) and (arm_vel > self.arm_thresh)
        
        if finger_active or arm_active:
            self.idle_counter = 0
            self.is_active = True
        else:
            self.idle_counter += 1
            if self.idle_counter >= self.idle_frames_req:
                self.is_active = False
                
        return self.is_active
