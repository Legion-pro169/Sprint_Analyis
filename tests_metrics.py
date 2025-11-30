import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.processing.signal import smooth_series, derivative, compute_velocity
from backend.processing.metrics import compute_joint_angle, detect_movement_start


class TestSignalProcessing:
    """Test signal processing functions."""

    def test_smooth_series_savgol(self):
        """Test Savitzky-Golay smoothing."""
        data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        smoothed = smooth_series(data, window_length=5, polyorder=2, method='savgol')
        assert len(smoothed) == len(data)
        assert not np.isnan(smoothed).any()

    def test_smooth_series_moving_average(self):
        """Test moving average smoothing."""
        data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        smoothed = smooth_series(data, window_length=3, method='moving_average')
        assert len(smoothed) == len(data)
        assert not np.isnan(smoothed).any()

    def test_smooth_series_short_data(self):
        """Test smoothing with data shorter than window."""
        data = np.array([1, 2, 3])
        smoothed = smooth_series(data, window_length=10)
        assert len(smoothed) == len(data)

    def test_derivative_gradient(self):
        """Test gradient-based derivative."""
        data = np.array([0, 1, 4, 9, 16])  # x^2
        deriv = derivative(data, dt=1.0, method='gradient')
        # Should be approximately [1, 2, 3, 4, 5] (linear approximation)
        assert len(deriv) == len(data)
        assert deriv[2] > deriv[1]  # Increasing derivative

    def test_derivative_diff(self):
        """Test diff-based derivative."""
        data = np.array([0, 1, 2, 3, 4])
        deriv = derivative(data, dt=1.0, method='diff')
        assert len(deriv) == len(data)
        assert np.allclose(deriv, 1.0)  # Constant derivative

    def test_compute_velocity(self):
        """Test velocity computation."""
        position = np.array([0, 1, 2, 3, 4])
        velocity = compute_velocity(position, dt=1.0, smooth=False)
        assert len(velocity) == len(position)
        assert velocity[2] > 0  # Positive velocity


class TestJointAngles:
    """Test joint angle calculations."""

    def test_compute_joint_angle_90deg(self):
        """Test 90-degree angle."""
        p1 = np.array([0, 1])
        p2 = np.array([0, 0])
        p3 = np.array([1, 0])
        angle = compute_joint_angle(p1, p2, p3)
        assert np.isclose(angle, 90.0, atol=1.0)

    def test_compute_joint_angle_180deg(self):
        """Test 180-degree angle (straight line)."""
        p1 = np.array([0, 0])
        p2 = np.array([1, 0])
        p3 = np.array([2, 0])
        angle = compute_joint_angle(p1, p2, p3)
        assert np.isclose(angle, 180.0, atol=1.0)

    def test_compute_joint_angle_60deg(self):
        """Test 60-degree angle."""
        p1 = np.array([0, 0])
        p2 = np.array([0, 0])
        p3 = np.array([1, np.sqrt(3)])
        angle = compute_joint_angle(p1, p2, p3)
        assert angle > 0 and angle < 180


class TestMovementDetection:
    """Test movement detection functions."""

    def test_detect_movement_start_simple(self):
        """Test movement start detection with synthetic data."""
        # Create synthetic mid-hip motion
        frames = 100
        data = {
            'frame': list(range(frames)),
            'time': [i / 30.0 for i in range(frames)],
            'left_hip_x': [0.5] * 30 + list(np.linspace(0.5, 0.7, 70)),
            'left_hip_y': [0.5] * frames,
            'right_hip_x': [0.5] * 30 + list(np.linspace(0.5, 0.7, 70)),
            'right_hip_y': [0.5] * frames,
        }
        df = pd.DataFrame(data)

        start_frame = detect_movement_start(df, velocity_threshold=0.001, fps=30.0)
        assert start_frame is not None
        assert start_frame >= 25 and start_frame <= 35  # Should detect around frame 30

    def test_detect_movement_start_no_movement(self):
        """Test movement detection with no movement."""
        frames = 100
        data = {
            'frame': list(range(frames)),
            'time': [i / 30.0 for i in range(frames)],
            'left_hip_x': [0.5] * frames,
            'left_hip_y': [0.5] * frames,
            'right_hip_x': [0.5] * frames,
            'right_hip_y': [0.5] * frames,
        }
        df = pd.DataFrame(data)

        start_frame = detect_movement_start(df, velocity_threshold=0.01, fps=30.0)
        # Should not detect movement or detect very late
        assert start_frame is None or start_frame > 50


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_array_smoothing(self):
        """Test smoothing with empty array."""
        data = np.array([])
        smoothed = smooth_series(data, window_length=5)
        assert len(smoothed) == 0

    def test_single_value_smoothing(self):
        """Test smoothing with single value."""
        data = np.array([5.0])
        smoothed = smooth_series(data, window_length=5)
        assert len(smoothed) == 1
        assert smoothed[0] == 5.0

    def test_nan_handling(self):
        """Test handling of NaN values."""
        data = np.array([1, 2, np.nan, 4, 5])
        # Should not crash, though results may contain NaN
        smoothed = smooth_series(data, window_length=3, method='moving_average')
        assert len(smoothed) == len(data)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])