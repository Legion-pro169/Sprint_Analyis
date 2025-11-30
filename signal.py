import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from typing import Optional


def smooth_series(data: np.ndarray,
                  window_length: int = 5,
                  polyorder: int = 2,
                  method: str = 'savgol') -> np.ndarray:
    """
    Smooth 1D time series data.

    Args:
        data: Input array
        window_length: Window size for smoothing (must be odd)
        polyorder: Polynomial order for Savitzky-Golay
        method: 'savgol' or 'moving_average'

    Returns:
        Smoothed array
    """
    if len(data) < window_length:
        return data

    if window_length % 2 == 0:
        window_length += 1

    if method == 'savgol':
        if window_length < polyorder + 2:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        return savgol_filter(data, window_length, polyorder)

    elif method == 'moving_average':
        kernel = np.ones(window_length) / window_length
        return np.convolve(data, kernel, mode='same')

    else:
        raise ValueError(f"Unknown method: {method}")


def lowpass_filter(data: np.ndarray,
                   cutoff_freq: float,
                   sampling_rate: float,
                   order: int = 4) -> np.ndarray:
    """
    Apply Butterworth lowpass filter.

    Args:
        data: Input array
        cutoff_freq: Cutoff frequency in Hz
        sampling_rate: Sampling rate in Hz
        order: Filter order

    Returns:
        Filtered array
    """
    if len(data) < 3 * order:
        return data

    nyquist = sampling_rate / 2.0
    normal_cutoff = cutoff_freq / nyquist

    if normal_cutoff >= 1.0:
        return data

    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def derivative(data: np.ndarray,
               dt: float = 1.0,
               method: str = 'gradient') -> np.ndarray:
    """
    Compute derivative of time series.

    Args:
        data: Input array
        dt: Time step
        method: 'gradient' or 'diff'

    Returns:
        Derivative array
    """
    if method == 'gradient':
        return np.gradient(data, dt)
    elif method == 'diff':
        deriv = np.diff(data) / dt
        return np.concatenate([[deriv[0]], deriv])
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_velocity(position: np.ndarray,
                     dt: float,
                     smooth: bool = True,
                     window_length: int = 5) -> np.ndarray:
    """
    Compute velocity from position data.

    Args:
        position: Position time series
        dt: Time step
        smooth: Whether to smooth result
        window_length: Smoothing window

    Returns:
        Velocity array
    """
    vel = derivative(position, dt, method='gradient')

    if smooth:
        vel = smooth_series(vel, window_length=window_length)

    return vel


def compute_acceleration(velocity: np.ndarray,
                         dt: float,
                         smooth: bool = True,
                         window_length: int = 5) -> np.ndarray:
    """
    Compute acceleration from velocity data.

    Args:
        velocity: Velocity time series
        dt: Time step
        smooth: Whether to smooth result
        window_length: Smoothing window

    Returns:
        Acceleration array
    """
    acc = derivative(velocity, dt, method='gradient')

    if smooth:
        acc = smooth_series(acc, window_length=window_length)

    return acc