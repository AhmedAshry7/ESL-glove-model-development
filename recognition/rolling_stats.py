import numpy as np

# Channel index groups used for arm activity indicators
_RIGHT_FINGER_IDX = list(range(0, 16))
_LEFT_FINGER_IDX  = list(range(28, 44))
_RIGHT_IMU_IDX    = list(range(16, 28))
_LEFT_IMU_IDX     = list(range(44, 56))

class RollingStatsBuffer:
    """
    Maintains a circular buffer of recent frames and computes
    statistical features for the random forest pre-filter.

    Feature vector layout (Option D — 282 features total):
      [  0: 56] — current pose snapshot (last frame)
      [ 56:112] — velocity (last frame − second-to-last frame)
      [112:280] — 3 sub-window energies per channel (56 ch × 3 windows)
      [280:282] — arm activity: (right_energy_magnitude, left_energy_magnitude)
    """
    def __init__(self, window_size=50, n_features=56):
        self.window_size = window_size
        self.n_features = n_features
        self.buffer = np.zeros((window_size, n_features))
        self.idx = 0
        self.full = False

    @property
    def n_frames(self) -> int:
        """Number of valid frames currently held."""
        return self.window_size if self.full else self.idx

    @property
    def fill_ratio(self) -> float:
        return self.n_frames / self.window_size

    def push(self, frame: np.ndarray):
        """Add a new frame to the circular buffer."""
        self.buffer[self.idx % self.window_size] = frame
        self.idx += 1
        if self.idx >= self.window_size:
            self.full = True

    def get_data(self) -> np.ndarray:
        """Returns the current valid data in chronological order."""
        if not self.full:
            return self.buffer[:self.idx]
        else:
            current_pos = self.idx % self.window_size
            return np.vstack((self.buffer[current_pos:], self.buffer[:current_pos]))

    def get_stats_features(self) -> np.ndarray:
        """
        Extract fixed-length Option-D feature vector from current buffer.

        Returns None if fewer than 5 frames are available.
        Returns shape (282,).
        """
        data = self.get_data()
        n = len(data)
        if n < 5:
            return None

        features = []

        # ── 1. Current pose snapshot (last frame) ── 56 values
        features.append(data[-1])

        # ── 2. Velocity (last − second-to-last) ── 56 values
        features.append(data[-1] - data[-2])

        # ── 3. Sub-window energies (sum of |diff|) ── 56 × 3 = 168 values
        # Divide the available frames into 3 equal chunks; each chunk's
        # energy describes how much motion occurred in that temporal slice.
        chunk_size = max(2, n // 3)
        boundaries = [
            (0,            chunk_size),
            (chunk_size,   2 * chunk_size),
            (2 * chunk_size, n),
        ]
        for start, end in boundaries:
            segment = data[start:end]
            if len(segment) < 2:
                # Pad with zeros if a segment is degenerate
                features.append(np.zeros(self.n_features))
            else:
                energy = np.sum(np.abs(np.diff(segment, axis=0)), axis=0)
                features.append(energy)

        # ── 4. Arm activity indicators ── 2 values
        # Total energy (sum |diff|) in right-arm channels vs. left-arm channels.
        # This helps the RF immediately distinguish left-hand from right-hand signs.
        diffs = np.abs(np.diff(data, axis=0))
        right_channels = _RIGHT_FINGER_IDX + _RIGHT_IMU_IDX
        left_channels  = _LEFT_FINGER_IDX  + _LEFT_IMU_IDX
        right_activity = np.sum(diffs[:, right_channels])
        left_activity  = np.sum(diffs[:, left_channels])
        features.append(np.array([right_activity, left_activity]))

        return np.concatenate(features)   # (282,)
