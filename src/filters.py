"""
ECG Filtering Module
Bandpass filtering and baseline wander removal

Theory:
- AHA recommends 0.5-100 Hz for monitoring ECG
- Zero-phase filtering (filtfilt) avoids phase distortion
- Butterworth filter has maximally flat passband
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = 0.5,
    highcut: float = 100.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter.

    Args:
        signal: Input signal (1D array)
        fs: Sampling frequency (Hz)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal

    Theory:
        Butterworth filter has maximally flat frequency response.
        filtfilt applies filter forward and backward for zero phase.
    """
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq

    # Ensure frequencies are within valid range
    low = max(low, 0.001)
    high = min(high, 0.999)

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal, axis=0)


def highpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff: float = 0.5,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth high-pass filter for baseline wander removal.

    Args:
        signal: Input signal
        fs: Sampling frequency (Hz)
        cutoff: Cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal (baseline removed)

    Theory:
        Baseline wander is caused by respiration and movement.
        Typically < 0.5 Hz. High-pass at 0.5 Hz removes it
        while preserving P-wave and T-wave morphology.
    """
    nyq = fs / 2
    normalized_cutoff = cutoff / nyq

    b, a = butter(order, normalized_cutoff, btype='high')
    return filtfilt(b, a, signal, axis=0)


def lowpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff: float = 100.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter for high-frequency noise removal.

    Args:
        signal: Input signal
        fs: Sampling frequency (Hz)
        cutoff: Cutoff frequency (Hz)
        order: Filter order

    Returns:
        Filtered signal

    Theory:
        Muscle noise (EMG) and other artifacts are typically > 100 Hz.
        Low-pass filtering removes these while preserving ECG components.
    """
    nyq = fs / 2
    normalized_cutoff = cutoff / nyq

    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, signal, axis=0)


def notch_filter(
    signal: np.ndarray,
    fs: float,
    freq: float = 50.0,
    quality: float = 30.0
) -> np.ndarray:
    """
    Apply notch filter to remove powerline interference.

    Args:
        signal: Input signal
        fs: Sampling frequency (Hz)
        freq: Notch frequency (50 Hz Europe, 60 Hz USA)
        quality: Quality factor (higher = narrower notch)

    Returns:
        Filtered signal

    Theory:
        Powerline interference appears at 50/60 Hz.
        Notch filter removes specific frequency with minimal
        impact on surrounding frequencies.
    """
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, signal, axis=0)


def remove_baseline_median(
    signal: np.ndarray,
    fs: float,
    window_ms: float = 600.0
) -> np.ndarray:
    """
    Remove baseline using median filter (two-pass).

    Args:
        signal: Input signal
        fs: Sampling frequency (Hz)
        window_ms: Window size in milliseconds

    Returns:
        Signal with baseline removed

    Theory:
        From PSB_master/baseline_ECG.m:
        Two-pass median filter preserves sharp QRS edges
        while removing slow baseline drift.
        - First pass: 200ms window
        - Second pass: 600ms window
    """
    from scipy.ndimage import median_filter

    # First pass: 200ms
    window1 = int(0.2 * fs)
    if window1 % 2 == 0:
        window1 += 1  # Must be odd

    baseline1 = median_filter(signal, size=window1, mode='reflect')

    # Second pass: 600ms
    window2 = int(window_ms / 1000 * fs)
    if window2 % 2 == 0:
        window2 += 1

    baseline2 = median_filter(baseline1, size=window2, mode='reflect')

    return signal - baseline2
