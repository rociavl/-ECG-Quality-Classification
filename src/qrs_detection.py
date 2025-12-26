"""
QRS Detection Module

Provides multiple QRS detection methods:
- Pan-Tompkins algorithm (pan_tompkins.py)
- NeuroKit2 wrapper
- Simple peak detection

Based on:
- PSB_master/pan_tompkin.m
- PSB_master/nqrsdetect.m (Afonso et al. 1999)
"""

import numpy as np
from dataclasses import dataclass

from .pan_tompkins import pan_tompkins, detect_qrs_pan_tompkins


@dataclass
class QRSResult:
    """Results from QRS detection."""
    r_peaks: np.ndarray        # R-peak indices
    rr_intervals: np.ndarray   # RR intervals in seconds
    heart_rate: float          # Mean heart rate (bpm)
    rr_std: float              # RR standard deviation (s)
    rr_cv: float               # Coefficient of variation (std/mean)
    method: str                # Detection method used


def detect_r_peaks(
    signal: np.ndarray,
    fs: float,
    method: str = 'hybrid'
) -> np.ndarray:
    """
    Detect R-peaks using specified method.

    Args:
        signal: ECG signal (1D, single lead)
        fs: Sampling frequency (Hz)
        method: 'pan_tompkins', 'neurokit', or 'simple'

    Returns:
        Array of R-peak indices

    Methods:
        - hybrid: Pan-Tompkins first, fallback to NeuroKit if <5 peaks (default)
        - pan_tompkins: Our implementation based on Pan & Tompkins 1985
        - neurokit: NeuroKit2 library (if installed)
        - simple: Basic peak detection with scipy
    """
    signal = np.asarray(signal).flatten()
    duration = len(signal) / fs

    if method == 'hybrid':
        # Try Pan-Tompkins first
        peaks = detect_qrs_pan_tompkins(signal, fs)

        # If too few peaks for duration, fallback to NeuroKit
        min_expected = int(duration * 40 / 60)  # 40 bpm minimum
        if len(peaks) < min_expected:
            try:
                import neurokit2 as nk
                _, info = nk.ecg_peaks(signal, sampling_rate=int(fs))
                peaks = np.array(info['ECG_R_Peaks'])
            except Exception:
                pass  # Keep Pan-Tompkins result

        return peaks

    elif method == 'pan_tompkins':
        return detect_qrs_pan_tompkins(signal, fs)

    elif method == 'neurokit':
        try:
            import neurokit2 as nk
            _, info = nk.ecg_peaks(signal, sampling_rate=int(fs))
            return np.array(info['ECG_R_Peaks'])
        except Exception:
            # Fallback to pan_tompkins
            return detect_qrs_pan_tompkins(signal, fs)

    elif method == 'simple':
        from scipy.signal import find_peaks, butter, filtfilt

        # Bandpass filter 5-15 Hz
        nyq = fs / 2
        b, a = butter(3, [5 / nyq, 15 / nyq], btype='band')
        filtered = filtfilt(b, a, signal)

        # Find peaks with minimum distance (refractory period)
        min_distance = int(0.2 * fs)  # 200ms
        peaks, _ = find_peaks(
            np.abs(filtered),
            distance=min_distance,
            height=np.std(filtered)
        )
        return peaks

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_rr_intervals(r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """
    Compute RR intervals from R-peak locations.

    Args:
        r_peaks: R-peak indices
        fs: Sampling frequency (Hz)

    Returns:
        RR intervals in seconds

    Theory:
        RR interval = time between consecutive R-peaks
        Normal range: 0.6-1.2 seconds (50-100 bpm)
    """
    if len(r_peaks) < 2:
        return np.array([])

    rr_samples = np.diff(r_peaks)
    rr_seconds = rr_samples / fs

    return rr_seconds


def compute_heart_rate(rr_intervals: np.ndarray) -> float:
    """
    Compute mean heart rate from RR intervals.

    Args:
        rr_intervals: RR intervals in seconds

    Returns:
        Heart rate in beats per minute (bpm)

    Theory:
        HR (bpm) = 60 / mean(RR interval in seconds)
    """
    if len(rr_intervals) == 0:
        return 0.0

    # Filter physiologically plausible RR intervals (0.3-2.0 s = 30-200 bpm)
    valid_rr = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]

    if len(valid_rr) == 0:
        return 0.0

    mean_rr = np.mean(valid_rr)
    return 60.0 / mean_rr


def analyze_qrs(
    signal: np.ndarray,
    fs: float,
    method: str = 'pan_tompkins'
) -> QRSResult:
    """
    Complete QRS analysis: detection + RR analysis.

    Args:
        signal: ECG signal (1D, single lead)
        fs: Sampling frequency (Hz)
        method: Detection method ('pan_tompkins', 'neurokit', 'simple')

    Returns:
        QRSResult with peaks, RR intervals, HR, and variability metrics

    Theory:
        CV (coefficient of variation) indicates rhythm regularity:
        - CV < 0.1: Very regular (normal sinus rhythm)
        - CV 0.1-0.2: Normal heart rate variability
        - CV > 0.3: Irregular (possible arrhythmia or artifact)
    """
    # Detect R-peaks
    r_peaks = detect_r_peaks(signal, fs, method)

    # Compute RR intervals
    rr_intervals = compute_rr_intervals(r_peaks, fs)

    # Compute statistics
    heart_rate = compute_heart_rate(rr_intervals)

    if len(rr_intervals) > 1:
        rr_std = np.std(rr_intervals)
        rr_mean = np.mean(rr_intervals)
        rr_cv = rr_std / rr_mean if rr_mean > 0 else 0.0
    else:
        rr_std = 0.0
        rr_cv = 0.0

    return QRSResult(
        r_peaks=r_peaks,
        rr_intervals=rr_intervals,
        heart_rate=heart_rate,
        rr_std=rr_std,
        rr_cv=rr_cv,
        method=method
    )


def get_expected_beats(duration_sec: float, hr_range: tuple = (40, 180)) -> tuple:
    """
    Calculate expected beat count range for a given duration.

    Args:
        duration_sec: Signal duration in seconds
        hr_range: (min_hr, max_hr) in bpm

    Returns:
        (min_beats, max_beats) expected

    Theory:
        For quality assessment:
        - Too few beats: Lead-off, flatline, or very low HR
        - Too many beats: Noise detected as beats, or tachycardia
        - 10s recording with normal HR (60-100 bpm): expect 10-17 beats
    """
    min_beats = int(duration_sec * hr_range[0] / 60)
    max_beats = int(duration_sec * hr_range[1] / 60)

    return (min_beats, max_beats)
