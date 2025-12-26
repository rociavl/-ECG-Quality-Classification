"""
ECG Preprocessing Pipeline
Combines filtering and QRS detection into a unified pipeline

This module imports from:
- filters.py: Bandpass, highpass, notch filtering, baseline removal
- qrs_detection.py: R-peak detection, RR intervals
- pan_tompkins.py: Pan-Tompkins QRS detection algorithm
- cardiac_frequency.py: Fourier/Burg HR estimation

Based on MATLAB implementations from:
- PSB_master/pan_tompkin.m
- PSB_master/baseline_ECG.m
- PSB_master/estimate_cardiac_frequency.m (Rocío Ávalos)
"""

import numpy as np
from dataclasses import dataclass

from .filters import (
    bandpass_filter,
    highpass_filter,
    notch_filter,
    remove_baseline_median
)
from .qrs_detection import (
    analyze_qrs,
    QRSResult
)
from .pan_tompkins import pan_tompkins, PanTompkinsResult
from .cardiac_frequency import (
    estimate_cardiac_frequency,
    CardiacFrequencyResult
)


@dataclass
class PreprocessedECG:
    """Container for preprocessed ECG data."""
    raw: np.ndarray              # Original signal
    filtered: np.ndarray         # After bandpass filtering
    fs: float                    # Sampling frequency
    qrs: QRSResult | None        # QRS analysis results (per lead)
    lead_name: str               # Lead identifier


def preprocess_single_lead(
    signal: np.ndarray,
    fs: float,
    lead_name: str = '',
    lowcut: float = 0.5,
    highcut: float = 100.0,
    notch_freq: float | None = None,
    detect_qrs: bool = True
) -> PreprocessedECG:
    """
    Preprocess a single ECG lead.

    Args:
        signal: Raw ECG signal (1D)
        fs: Sampling frequency (Hz)
        lead_name: Lead identifier (e.g., 'II')
        lowcut: High-pass cutoff (Hz)
        highcut: Low-pass cutoff (Hz)
        notch_freq: Powerline frequency (50/60 Hz) or None to skip
        detect_qrs: Whether to perform QRS detection

    Returns:
        PreprocessedECG with filtered signal and QRS results

    Pipeline:
        1. Bandpass filter (0.5-100 Hz)
        2. Optional notch filter (50/60 Hz)
        3. QRS detection (if enabled)
    """
    # Step 1: Bandpass filter
    filtered = bandpass_filter(signal, fs, lowcut, highcut)

    # Step 2: Optional notch filter
    if notch_freq is not None:
        filtered = notch_filter(filtered, fs, notch_freq)

    # Step 3: QRS detection
    qrs = None
    if detect_qrs:
        try:
            qrs = analyze_qrs(filtered, fs)
        except Exception:
            qrs = None

    return PreprocessedECG(
        raw=signal,
        filtered=filtered,
        fs=fs,
        qrs=qrs,
        lead_name=lead_name
    )


def preprocess_multilead(
    signal: np.ndarray,
    fs: float,
    lead_names: list[str],
    reference_lead: str = 'II',
    **kwargs
) -> dict[str, PreprocessedECG]:
    """
    Preprocess all leads of a 12-lead ECG.

    Args:
        signal: Multi-lead ECG (samples, leads)
        fs: Sampling frequency (Hz)
        lead_names: List of lead names
        reference_lead: Lead for QRS detection (typically 'II')
        **kwargs: Additional arguments for preprocess_single_lead

    Returns:
        Dictionary mapping lead names to PreprocessedECG

    Theory:
        - Filter all leads identically
        - QRS detection on reference lead only (usually Lead II)
        - Lead II has best P-wave and QRS visibility
    """
    results = {}

    for i, lead_name in enumerate(lead_names):
        lead_signal = signal[:, i]

        # Only detect QRS on reference lead
        detect_qrs = (lead_name == reference_lead)

        results[lead_name] = preprocess_single_lead(
            lead_signal,
            fs,
            lead_name,
            detect_qrs=detect_qrs,
            **kwargs
        )

    return results


def preprocess_ecg(
    signal: np.ndarray,
    fs: float,
    lead_names: list[str] | None = None
) -> dict[str, PreprocessedECG]:
    """
    Main preprocessing function for ECG records.

    Args:
        signal: ECG signal (samples,) or (samples, leads)
        fs: Sampling frequency (Hz)
        lead_names: Lead names (optional)

    Returns:
        Dictionary of preprocessed leads

    Usage:
        from src.preprocessing import preprocess_ecg
        from src.data_loader import ECGDataset

        dataset = ECGDataset('data/challenge2011/set-a')
        ecg = dataset[0]

        processed = preprocess_ecg(ecg.signal, ecg.fs, ecg.leads)
        lead_ii = processed['II']

        print(f"Heart rate: {lead_ii.qrs.heart_rate:.1f} bpm")
    """
    # Handle 1D signal
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
        lead_names = lead_names or ['Lead']

    # Default lead names
    if lead_names is None:
        lead_names = [f'Lead_{i}' for i in range(signal.shape[1])]

    # Determine reference lead
    if 'II' in lead_names:
        reference_lead = 'II'
    else:
        reference_lead = lead_names[0]

    return preprocess_multilead(
        signal,
        fs,
        lead_names,
        reference_lead=reference_lead
    )


def get_filtered_signal(
    signal: np.ndarray,
    fs: float,
    method: str = 'bandpass'
) -> np.ndarray:
    """
    Quick filtering without full preprocessing.

    Args:
        signal: ECG signal
        fs: Sampling frequency (Hz)
        method: 'bandpass', 'highpass', or 'median'

    Returns:
        Filtered signal

    Use this for quick filtering without QRS detection.
    """
    if method == 'bandpass':
        return bandpass_filter(signal, fs)
    elif method == 'highpass':
        return highpass_filter(signal, fs)
    elif method == 'median':
        return remove_baseline_median(signal, fs)
    else:
        raise ValueError(f"Unknown method: {method}")
