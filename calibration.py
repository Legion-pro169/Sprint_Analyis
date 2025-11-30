import numpy as np
from typing import Optional, Tuple


class Calibrator:
    """Handle pixel-to-meter calibration."""

    def __init__(self,
                 athlete_height_m: Optional[float] = None,
                 measured_height_px: Optional[float] = None):
        """
        Initialize calibrator.

        Args:
            athlete_height_m: Known athlete height in meters
            measured_height_px: Measured height in pixels from video
        """
        self.athlete_height_m = athlete_height_m
        self.measured_height_px = measured_height_px
        self.scale_factor = None

        if athlete_height_m and measured_height_px:
            self.scale_factor = athlete_height_m / measured_height_px

    def px_to_m(self, value_px: float) -> float:
        """
        Convert pixels to meters.

        Args:
            value_px: Value in pixels

        Returns:
            Value in meters (or pixels if not calibrated)
        """
        if self.scale_factor:
            return value_px * self.scale_factor
        return value_px

    def estimate_height_from_landmarks(self, landmarks: np.ndarray) -> float:
        """
        Estimate person height in pixels from landmarks.
        Approximates as distance from nose to ankle.

        Args:
            landmarks: Nx4 array of (x, y, z, visibility)

        Returns:
            Estimated height in pixels
        """
        # MediaPipe landmarks: 0=nose, 27=left_ankle, 28=right_ankle
        nose_idx = 0
        left_ankle_idx = 27
        right_ankle_idx = 28

        nose = landmarks[nose_idx, :2]
        left_ankle = landmarks[left_ankle_idx, :2]
        right_ankle = landmarks[right_ankle_idx, :2]

        ankle = (left_ankle + right_ankle) / 2

        height_px = np.linalg.norm(nose - ankle)
        return height_px

    def auto_calibrate(self,
                       landmarks_list: list,
                       athlete_height_m: float,
                       frame_width: int,
                       frame_height: int) -> bool:
        """
        Auto-calibrate using landmarks from multiple frames.

        Args:
            landmarks_list: List of landmark arrays
            athlete_height_m: Known athlete height in meters
            frame_width: Video frame width
            frame_height: Video frame height

        Returns:
            Success status
        """
        heights = []

        for landmarks in landmarks_list:
            if landmarks is None or len(landmarks) == 0:
                continue

            # Convert normalized to pixel coordinates
            lm_px = landmarks.copy()
            lm_px[:, 0] *= frame_width
            lm_px[:, 1] *= frame_height

            h = self.estimate_height_from_landmarks(lm_px)
            if h > 0:
                heights.append(h)

        if heights:
            median_height_px = np.median(heights)
            self.measured_height_px = median_height_px
            self.athlete_height_m = athlete_height_m
            self.scale_factor = athlete_height_m / median_height_px
            return True

        return False


def apply_homography_stub(points: np.ndarray,
                          matrix: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Stub for homography transformation.

    Args:
        points: Nx2 array of points
        matrix: 3x3 homography matrix (optional)

    Returns:
        Transformed points (identity if matrix is None)
    """
    if matrix is None:
        return points

    # Apply homography transformation
    points_h = np.column_stack([points, np.ones(len(points))])
    transformed = (matrix @ points_h.T).T
    transformed = transformed[:, :2] / transformed[:, 2:3]

    return transformed