"""
PSD Parameter Extraction Module

Complete spectral parameters from psdpar.m (Abel Torres, IBEC, 2024).
Reference: BSA Course - PSD Parameter Estimation

These parameters characterize the power spectral density distribution
and are useful for ECG quality assessment.
"""

import numpy as np
from typing import Dict


def compute_psd_parameters(f: np.ndarray, Pxx: np.ndarray) -> Dict[str, float]:
    """
    Compute complete PSD parameters from frequency and power arrays.

    Based on psdpar.m (Abel Torres, IBEC, 2024) from BSA course materials.

    Args:
        f: Frequency array (Hz) from periodogram/welch
        Pxx: Power spectral density array

    Returns:
        Dictionary with 12 spectral parameters (complete psdpar.m):
        - f_peak: Peak frequency (maximum PSD)
        - f_mean: Spectral centroid (mean frequency)
        - f_median: Median frequency (50% cumulative power)
        - f_q25: 25th percentile frequency
        - f_q75: 75th percentile frequency
        - f_max95: 95th percentile frequency (effective bandwidth)
        - f_std: Spectral spread (standard deviation)
        - f_iqr: Interquartile range (f_q75 - f_q25)
        - hShannon: Normalized Shannon entropy (0=impulse, 1=flat)
        - spectral_asymmetry: Spectral skewness (3rd moment)
        - spectral_kurtosis: Spectral kurtosis (4th moment, excess)
        - spectral_flatness: Wiener entropy (geometric/arithmetic mean)

    Theory:
        Good ECG: Concentrated power, low flatness, positive kurtosis
        Bad ECG: Spread power, high flatness, near-zero kurtosis
    """
    # Ensure arrays are numpy
    f = np.asarray(f)
    Pxx = np.asarray(Pxx)

    # Handle edge cases
    if len(f) == 0 or len(Pxx) == 0 or np.sum(Pxx) == 0:
        return _empty_parameters()

    # Peak frequency (maximum of the PSD)
    # fpeak = f at max(Pxx)
    peak_idx = np.argmax(Pxx)
    f_peak = f[peak_idx]

    # Normalize PSD to probability distribution
    total_power = np.sum(Pxx)
    Pxx_norm = Pxx / (total_power + 1e-10)

    # Cumulative power distribution
    cumsum_power = np.cumsum(Pxx_norm)

    # Quantile frequencies (robust frequency descriptors)
    f_q25 = _find_quantile_freq(f, cumsum_power, 0.25)
    f_median = _find_quantile_freq(f, cumsum_power, 0.50)
    f_q75 = _find_quantile_freq(f, cumsum_power, 0.75)
    f_max95 = _find_quantile_freq(f, cumsum_power, 0.95)

    # Spectral centroid (mean frequency)
    # fmean = Σ(f × Pxx) / Σ(Pxx)
    f_mean = np.sum(f * Pxx_norm)

    # Spectral spread (standard deviation)
    # fstd = √(Σ((f - fmean)² × Pxx) / Σ(Pxx))
    f_std = np.sqrt(np.sum((f - f_mean)**2 * Pxx_norm))

    # Interquartile range
    f_iqr = f_q75 - f_q25

    # Normalized Shannon entropy (from psdpar.m)
    # h = -Σ(p * log(p)) for p > 0
    # hShannon = h / log(N)  (normalized to [0, 1])
    # 0 = impulse distribution (all power at one frequency)
    # 1 = flat distribution (uniform power)
    p = Pxx_norm[Pxx_norm > 0]
    if len(p) > 1:
        h = -np.sum(p * np.log(p))
        hShannon = h / np.log(len(Pxx_norm))
        hShannon = np.clip(hShannon, 0.0, 1.0)
    else:
        hShannon = 0.0

    # Higher moments (spectral shape parameters)
    f_centered = f - f_mean

    # Spectral asymmetry (skewness) - c_Asymmetry = μ₃/σ³
    # Positive: Right-skewed (high freq tail)
    # Negative: Left-skewed (low freq tail)
    if f_std > 1e-10:
        spectral_asymmetry = np.sum(f_centered**3 * Pxx_norm) / (f_std**3)
        # Bounds check: typical range is -10 to +50 for ECG
        spectral_asymmetry = np.clip(spectral_asymmetry, -100, 100)
    else:
        spectral_asymmetry = 0.0

    # Spectral kurtosis (excess) - c_Kurtosis = μ₄/σ⁴ - 3
    # Positive: Heavy tails, sharp peaks (leptokurtic)
    # Negative: Light tails (platykurtic)
    # Zero: Gaussian-like
    if f_std > 1e-10:
        spectral_kurtosis = np.sum(f_centered**4 * Pxx_norm) / (f_std**4) - 3
        # Bounds check: typical range is -3 to +500 for ECG
        spectral_kurtosis = np.clip(spectral_kurtosis, -100, 1000)
    else:
        spectral_kurtosis = 0.0

    # Spectral flatness (Wiener entropy)
    # Ratio of geometric mean to arithmetic mean of PSD
    # Range: 0 (tonal/periodic signal) to 1 (white noise)
    # Good ECG: Low flatness (periodic QRS pattern)
    # Bad ECG: High flatness (noise-like)
    spectral_flatness = _compute_spectral_flatness(Pxx)

    return {
        'f_peak': float(f_peak),
        'f_mean': float(f_mean),
        'f_median': float(f_median),
        'f_q25': float(f_q25),
        'f_q75': float(f_q75),
        'f_max95': float(f_max95),
        'f_std': float(f_std),
        'f_iqr': float(f_iqr),
        'hShannon': float(hShannon),
        'spectral_asymmetry': float(spectral_asymmetry),
        'spectral_kurtosis': float(spectral_kurtosis),
        'spectral_flatness': float(spectral_flatness)
    }


def _find_quantile_freq(f: np.ndarray, cumsum: np.ndarray, quantile: float) -> float:
    """Find frequency at given quantile of cumulative power."""
    idx = np.searchsorted(cumsum, quantile)
    if idx >= len(f):
        idx = len(f) - 1
    return f[idx]


def _compute_spectral_flatness(Pxx: np.ndarray) -> float:
    """
    Compute spectral flatness (Wiener entropy).

    Spectral flatness = geometric_mean(Pxx) / arithmetic_mean(Pxx)

    A value of 1 indicates white noise (flat spectrum).
    A value close to 0 indicates a tonal/periodic signal.
    """
    Pxx_positive = Pxx[Pxx > 0]

    if len(Pxx_positive) == 0:
        return 0.0

    # Geometric mean via log domain (numerically stable)
    log_mean = np.mean(np.log(Pxx_positive + 1e-10))
    geom_mean = np.exp(log_mean)

    # Arithmetic mean
    arith_mean = np.mean(Pxx_positive)

    if arith_mean < 1e-10:
        return 0.0

    return geom_mean / arith_mean


def _empty_parameters() -> Dict[str, float]:
    """Return dictionary with zero values for all parameters."""
    return {
        'f_peak': 0.0,
        'f_mean': 0.0,
        'f_median': 0.0,
        'f_q25': 0.0,
        'f_q75': 0.0,
        'f_max95': 0.0,
        'f_std': 0.0,
        'f_iqr': 0.0,
        'hShannon': 0.0,
        'spectral_asymmetry': 0.0,
        'spectral_kurtosis': 0.0,
        'spectral_flatness': 0.0
    }
