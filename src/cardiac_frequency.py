"""
Cardiac Frequency Estimation

Python implementation based on:
- PSB_master/estimate_cardiac_frequency.m (Rocío Ávalos, Antonio Di Pierro)
- BSA Lab 3, November 2025

Methods:
    - Fourier: Periodogram with robust peak detection
    - Burg: AR model-based pole estimation
"""

import numpy as np
from scipy.signal import periodogram, find_peaks, detrend

from .ar_order import select_ar_order
from dataclasses import dataclass


# Physiological constraints
FC_MIN = 0.5   # 30 bpm minimum
FC_MAX = 2.5   # 150 bpm maximum


@dataclass
class CardiacFrequencyResult:
    """Results from cardiac frequency estimation."""
    fcard: float         # Cardiac frequency (Hz)
    hr: float            # Heart rate (bpm)
    method: str          # Method used
    confidence: float    # Confidence score (0-1)
    psd: np.ndarray | None      # PSD (for Fourier method)
    frequencies: np.ndarray | None  # Frequency axis


def estimate_cardiac_frequency_fourier(
    ecg_signal: np.ndarray,
    fs: float,
    nfft: int | None = None
) -> CardiacFrequencyResult:
    """
    Estimate cardiac frequency using Fourier (periodogram) method.

    Args:
        ecg_signal: ECG signal vector
        fs: Sampling frequency (Hz)
        nfft: FFT size (default: 2^max(12, nextpow2(N)))

    Returns:
        CardiacFrequencyResult with fcard and HR

    Theory:
        - Uses periodogram with Hamming window
        - Detects peaks in cardiac frequency range (0.5-2.5 Hz)
        - Prefers lowest frequency peak (fundamental)
        - Handles harmonic detection
    """
    # Detrend and vectorize
    ecg_signal = detrend(np.asarray(ecg_signal).flatten())
    N = len(ecg_signal)

    if N < 20:
        return CardiacFrequencyResult(
            fcard=np.nan, hr=np.nan, method='fourier',
            confidence=0.0, psd=None, frequencies=None
        )

    # FFT size
    if nfft is None:
        nfft = 2 ** max(12, int(np.ceil(np.log2(N))))

    # Compute periodogram with Hamming window
    f, Pxx = periodogram(ecg_signal, fs, window='hamming', nfft=nfft)

    # Extract cardiac frequency range
    freq_mask = (f >= FC_MIN) & (f <= FC_MAX)
    Pxx_range = Pxx[freq_mask]
    f_range = f[freq_mask]

    if len(Pxx_range) == 0:
        return CardiacFrequencyResult(
            fcard=np.nan, hr=np.nan, method='fourier',
            confidence=0.0, psd=Pxx, frequencies=f
        )

    # Detect significant peaks
    min_peak_height = 0.1 * np.max(Pxx_range)
    peaks, properties = find_peaks(
        Pxx_range,
        height=min_peak_height,
        distance=int(0.3 * len(f_range) / (FC_MAX - FC_MIN))  # ~0.3 Hz separation
    )

    if len(peaks) > 0:
        peak_freqs = f_range[peaks]
        peak_powers = Pxx_range[peaks]

        # Strategy: prefer LOWEST frequency peak (fundamental)
        idx_min = np.argmin(peak_freqs)
        fundamental_peak = peak_powers[idx_min]
        fundamental_freq = peak_freqs[idx_min]

        # Check if fundamental has at least 15% of maximum peak power
        if fundamental_peak >= 0.15 * np.max(peak_powers):
            fcard = fundamental_freq
            confidence = fundamental_peak / np.max(peak_powers)
        else:
            # Fundamental too weak - might be detecting harmonics
            fcard = fundamental_freq
            confidence = 0.5

            # Try to find fundamental by dividing strong peaks
            for i in range(len(peak_freqs)):
                # Check if this could be 2nd harmonic
                potential_fund = peak_freqs[i] / 2
                if FC_MIN <= potential_fund <= FC_MAX:
                    idx = np.argmin(np.abs(f_range - potential_fund))
                    if Pxx_range[idx] > 0.1 * np.max(Pxx_range):
                        fcard = potential_fund
                        confidence = 0.7
                        break

                # Check if this could be 3rd harmonic
                potential_fund = peak_freqs[i] / 3
                if FC_MIN <= potential_fund <= FC_MAX:
                    idx = np.argmin(np.abs(f_range - potential_fund))
                    if Pxx_range[idx] > 0.1 * np.max(Pxx_range):
                        fcard = potential_fund
                        confidence = 0.6
                        break
    else:
        # No peaks detected - use maximum in range
        idx_max = np.argmax(Pxx_range)
        fcard = f_range[idx_max]
        confidence = 0.3

    hr = fcard * 60

    # Final sanity check
    if hr < 30 or hr > 180:
        fcard = np.nan
        hr = np.nan
        confidence = 0.0

    return CardiacFrequencyResult(
        fcard=fcard,
        hr=hr,
        method='fourier',
        confidence=confidence,
        psd=Pxx,
        frequencies=f
    )


def estimate_cardiac_frequency_burg(
    ecg_signal: np.ndarray,
    fs: float,
    model_order: int | None = None,
    auto_order: bool = True
) -> CardiacFrequencyResult:
    """
    Estimate cardiac frequency using Burg (AR model) method.

    Args:
        ecg_signal: ECG signal vector
        fs: Sampling frequency (Hz)
        model_order: AR model order (default: 16)

    Returns:
        CardiacFrequencyResult with fcard and HR

    Theory:
        - Uses Burg method to estimate AR coefficients
        - Finds poles (roots of AR polynomial)
        - Pole frequency = angle(pole) / pi * fs / 2
        - Only considers poles with positive imaginary part
    """
    from scipy.signal import lfilter

    # Auto-select model order if not provided (using AICm criterion)
    if model_order is None and auto_order:
        try:
            model_order, _ = select_ar_order(ecg_signal, max_order=30, criterion='aicm')
        except Exception:
            model_order = 16  # fallback default
    elif model_order is None:
        model_order = 16  # fallback default

    # Detrend and vectorize
    ecg_signal = detrend(np.asarray(ecg_signal).flatten())
    N = len(ecg_signal)

    if N < 10 * model_order:
        return CardiacFrequencyResult(
            fcard=np.nan, hr=np.nan, method='burg',
            confidence=0.0, psd=None, frequencies=None
        )

    try:
        # Burg method for AR coefficients
        ar_coeffs = _arburg(ecg_signal, model_order)

        # Find poles (roots of AR polynomial)
        poles = np.roots(ar_coeffs)

        # Only consider poles with POSITIVE imaginary part
        positive_freq_poles = poles[np.imag(poles) > 0]

        if len(positive_freq_poles) == 0:
            return CardiacFrequencyResult(
                fcard=np.nan, hr=np.nan, method='burg',
                confidence=0.0, psd=None, frequencies=None
            )

        # Calculate frequencies for all positive-frequency poles
        pole_frequencies = np.abs(np.angle(positive_freq_poles)) / np.pi * fs / 2
        pole_magnitudes = np.abs(positive_freq_poles)

        # Filter to physiological range
        valid_mask = (pole_frequencies >= FC_MIN) & (pole_frequencies <= FC_MAX)

        if not np.any(valid_mask):
            # Try harmonic correction
            fcard = np.nan
            for freq in pole_frequencies:
                for divisor in [2, 3, 4]:
                    fund_freq = freq / divisor
                    if FC_MIN <= fund_freq <= FC_MAX:
                        fcard = fund_freq
                        break
                if not np.isnan(fcard):
                    break

            if np.isnan(fcard):
                return CardiacFrequencyResult(
                    fcard=np.nan, hr=np.nan, method='burg',
                    confidence=0.0, psd=None, frequencies=None
                )
            confidence = 0.5
        else:
            # Select lowest frequency in valid range (fundamental)
            valid_frequencies = pole_frequencies[valid_mask]
            valid_magnitudes = pole_magnitudes[valid_mask]
            fcard = np.min(valid_frequencies)

            # Confidence based on pole magnitude (closer to unit circle = sharper peak)
            idx = np.argmin(valid_frequencies)
            confidence = valid_magnitudes[valid_mask][idx]

        hr = fcard * 60

        # Final sanity check
        if hr < 30 or hr > 180:
            fcard = np.nan
            hr = np.nan
            confidence = 0.0

        return CardiacFrequencyResult(
            fcard=fcard,
            hr=hr,
            method='burg',
            confidence=confidence,
            psd=None,
            frequencies=None
        )

    except Exception:
        return CardiacFrequencyResult(
            fcard=np.nan, hr=np.nan, method='burg',
            confidence=0.0, psd=None, frequencies=None
        )


def _arburg(x: np.ndarray, order: int) -> np.ndarray:
    """
    Burg method for AR parameter estimation.

    Args:
        x: Input signal
        order: Model order

    Returns:
        AR coefficients [1, a1, a2, ..., ap]
    """
    N = len(x)
    a = np.zeros(order + 1)
    a[0] = 1.0

    # Initialize forward and backward prediction errors
    ef = x.copy()
    eb = x.copy()

    for m in range(order):
        # Compute reflection coefficient
        efm = ef[m + 1:]
        ebm = eb[m:-1]

        num = -2.0 * np.dot(ebm, efm)
        den = np.dot(efm, efm) + np.dot(ebm, ebm)

        if den == 0:
            k = 0.0
        else:
            k = num / den

        # Update AR coefficients (Levinson recursion)
        a_new = np.zeros(m + 2)
        a_new[0] = 1.0
        for i in range(1, m + 2):
            if i <= m:
                a_new[i] = a[i] + k * a[m + 1 - i]
            else:
                a_new[i] = k

        a[:m + 2] = a_new

        # Update prediction errors
        ef_new = efm + k * ebm
        eb_new = ebm + k * efm
        ef[m + 1:] = ef_new
        eb[m + 1:] = eb_new

    return a


def estimate_cardiac_frequency(
    ecg_signal: np.ndarray,
    fs: float,
    method: str = 'fourier',
    **kwargs
) -> CardiacFrequencyResult:
    """
    Main interface for cardiac frequency estimation.

    Args:
        ecg_signal: ECG signal vector
        fs: Sampling frequency (Hz)
        method: 'fourier' or 'burg'
        **kwargs: Method-specific parameters

    Returns:
        CardiacFrequencyResult with fcard (Hz) and HR (bpm)
    """
    method = method.lower().strip()

    if method == 'fourier':
        return estimate_cardiac_frequency_fourier(ecg_signal, fs, **kwargs)
    elif method == 'burg':
        return estimate_cardiac_frequency_burg(ecg_signal, fs, **kwargs)
    else:
        raise ValueError(f"Method must be 'fourier' or 'burg', got '{method}'")
