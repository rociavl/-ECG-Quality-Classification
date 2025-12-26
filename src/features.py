"""
ECG Quality Feature Extraction Module

Features aligned with PhysioNet Challenge 2011 winning approaches:
- Xia et al. (spectral features)
- Li & Clifford (statistical moments)
- Hayn et al. (SNR-based)
- Kalkstein et al. (morphological)

BSA Course Requirement:
- Time-frequency features using STFT
"""

import warnings
import numpy as np
from scipy.signal import periodogram, spectrogram, butter, filtfilt
from dataclasses import dataclass

# Suppress numpy correlation warnings (we handle NaN values explicitly)
warnings.filterwarnings("ignore", message="invalid value encountered in divide")

from .qrs_detection import detect_r_peaks, compute_rr_intervals, compute_heart_rate
from .filters import bandpass_filter
from .spectral_params import compute_psd_parameters


# =============================================================================
# SNR FEATURES (Hayn et al., Johannesen)
# =============================================================================

def extract_snr_features(signal: np.ndarray, fs: float, r_peaks: np.ndarray | None = None) -> dict:
    """
    Extract Signal-to-Noise Ratio features.

    Based on Hayn et al. and Johannesen approaches from Challenge 2011.

    Args:
        signal: ECG signal (1D, single lead)
        fs: Sampling frequency (Hz)
        r_peaks: R-peak indices (optional, will detect if not provided)

    Returns:
        Dictionary with SNR features:
        - snr_db: Signal-to-noise ratio in dB
        - qrs_power: Power in QRS regions
        - noise_power: Power in non-QRS regions

    Theory:
        SNR = 10 * log10(signal_power / noise_power)
        Good ECG: SNR > 10 dB
        Bad ECG: SNR < 5 dB
    """
    signal = np.asarray(signal).flatten()

    # Detect R-peaks if not provided
    if r_peaks is None or len(r_peaks) == 0:
        r_peaks = detect_r_peaks(signal, fs)

    if len(r_peaks) < 2:
        return {
            'snr_db': 0.0,
            'qrs_power': 0.0,
            'noise_power': np.var(signal)
        }

    # Define QRS windows (100ms around each R-peak)
    qrs_width = int(0.1 * fs)  # 100ms
    qrs_mask = np.zeros(len(signal), dtype=bool)

    for peak in r_peaks:
        start = max(0, peak - qrs_width // 2)
        end = min(len(signal), peak + qrs_width // 2)
        qrs_mask[start:end] = True

    # Calculate powers
    qrs_signal = signal[qrs_mask]
    noise_signal = signal[~qrs_mask]

    qrs_power = np.var(qrs_signal) if len(qrs_signal) > 0 else 0.0
    noise_power = np.var(noise_signal) if len(noise_signal) > 0 else 1e-10

    # SNR in dB (with bounds check to avoid -inf)
    if qrs_power < 1e-10 or noise_power < 1e-10:
        snr_db = -20.0  # Very low SNR indicates failed/flat signal
    else:
        snr_db = 10 * np.log10(qrs_power / noise_power)
        snr_db = np.clip(snr_db, -20, 60)  # Reasonable range for ECG

    return {
        'snr_db': snr_db,
        'qrs_power': qrs_power,
        'noise_power': noise_power
    }


# =============================================================================
# BASELINE FEATURES (Ho & Chen)
# =============================================================================

def extract_baseline_features(signal: np.ndarray, fs: float) -> dict:
    """
    Extract baseline wander features.

    Based on Ho & Chen approach - baseline wander indicates poor electrode contact.

    Args:
        signal: ECG signal (1D)
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary with baseline features:
        - baseline_power: Power below 0.5 Hz
        - baseline_std: Standard deviation of baseline
        - baseline_range: Range of baseline drift

    Theory:
        Baseline wander is low-frequency drift (< 0.5 Hz)
        Caused by respiration, movement, poor electrode contact
        High baseline power = poor quality
    """
    signal = np.asarray(signal).flatten()

    # Extract baseline using low-pass filter at 0.5 Hz
    nyq = fs / 2
    if 0.5 / nyq < 1:
        b, a = butter(2, 0.5 / nyq, btype='low')
        baseline = filtfilt(b, a, signal)
    else:
        baseline = np.zeros_like(signal)

    # Calculate baseline statistics
    baseline_power = np.var(baseline)
    baseline_std = np.std(baseline)
    baseline_range = np.max(baseline) - np.min(baseline)

    # Relative baseline power (normalized by signal power)
    signal_power = np.var(signal)
    baseline_power_ratio = baseline_power / (signal_power + 1e-10)

    return {
        'baseline_power': baseline_power,
        'baseline_std': baseline_std,
        'baseline_range': baseline_range,
        'baseline_power_ratio': baseline_power_ratio
    }


# =============================================================================
# MORPHOLOGICAL FEATURES (Kalkstein et al.)
# =============================================================================

def extract_morphological_features(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray | None = None
) -> dict:
    """
    Extract morphological (beat-based) features.

    Based on Kalkstein et al. approach from Challenge 2011.

    Args:
        signal: ECG signal (1D)
        fs: Sampling frequency (Hz)
        r_peaks: R-peak indices (optional)

    Returns:
        Dictionary with morphological features:
        - beat_count: Number of detected beats
        - heart_rate: Mean heart rate (bpm)
        - rr_mean: Mean RR interval (s)
        - rr_std: RR interval standard deviation (s)
        - rr_cv: Coefficient of variation (regularity)

    Theory:
        - Normal 10s ECG: 8-17 beats (48-100 bpm)
        - CV < 0.1: Very regular rhythm
        - CV > 0.3: Irregular (artifact or arrhythmia)
    """
    signal = np.asarray(signal).flatten()
    duration = len(signal) / fs

    # Detect R-peaks if not provided
    if r_peaks is None:
        r_peaks = detect_r_peaks(signal, fs)

    beat_count = len(r_peaks)

    # Calculate RR intervals
    rr_intervals = compute_rr_intervals(r_peaks, fs)
    heart_rate = compute_heart_rate(rr_intervals)

    if len(rr_intervals) > 1:
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        rr_cv = rr_std / rr_mean if rr_mean > 0 else 0.0
    else:
        rr_mean = 0.0
        rr_std = 0.0
        rr_cv = 0.0

    # Expected beat count for 10s recording
    expected_min = int(duration * 40 / 60)  # 40 bpm
    expected_max = int(duration * 180 / 60)  # 180 bpm
    beat_count_valid = expected_min <= beat_count <= expected_max

    return {
        'beat_count': beat_count,
        'heart_rate': heart_rate,
        'rr_mean': rr_mean,
        'rr_std': rr_std,
        'rr_cv': rr_cv,
        'beat_count_valid': beat_count_valid
    }


# =============================================================================
# SPECTRAL FEATURES (Xia et al.)
# =============================================================================

def extract_spectral_features(signal: np.ndarray, fs: float) -> dict:
    """
    Extract frequency-domain features.

    Based on Xia et al. approach - spectral characteristics distinguish
    good ECG from noisy signals.

    Args:
        signal: ECG signal (1D)
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary with spectral features:
        - spectral_entropy: Entropy of power spectrum (0-1)
        - peak_freq: Dominant frequency (Hz)
        - cardiac_power_ratio: Power in cardiac band / total power
        - high_freq_power_ratio: Power > 40 Hz / total power

    Theory:
        - Good ECG: Most power in cardiac band (0.5-40 Hz)
        - Noisy ECG: High power at high frequencies (muscle noise)
        - Spectral entropy: Low for periodic signals, high for noise
    """
    signal = np.asarray(signal).flatten()

    # Remove DC component
    signal = signal - np.mean(signal)

    # Compute periodogram
    f, Pxx = periodogram(signal, fs, window='hamming')

    # Normalize to get probability distribution
    Pxx_norm = Pxx / (np.sum(Pxx) + 1e-10)

    # Spectral entropy (Shannon entropy)
    Pxx_nonzero = Pxx_norm[Pxx_norm > 0]
    spectral_entropy = -np.sum(Pxx_nonzero * np.log2(Pxx_nonzero))
    # Normalize to 0-1 range
    max_entropy = np.log2(len(Pxx_nonzero)) if len(Pxx_nonzero) > 0 else 1
    spectral_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0

    # Peak frequency (dominant frequency)
    peak_idx = np.argmax(Pxx)
    peak_freq = f[peak_idx]

    # Power in different bands
    total_power = np.sum(Pxx)

    # Cardiac band (0.5-40 Hz) - where ECG content should be
    cardiac_mask = (f >= 0.5) & (f <= 40)
    cardiac_power = np.sum(Pxx[cardiac_mask])
    cardiac_power_ratio = cardiac_power / (total_power + 1e-10)

    # High frequency band (> 40 Hz) - muscle noise
    hf_mask = f > 40
    hf_power = np.sum(Pxx[hf_mask])
    high_freq_power_ratio = hf_power / (total_power + 1e-10)

    # Very low frequency (< 0.5 Hz) - baseline wander
    vlf_mask = f < 0.5
    vlf_power = np.sum(Pxx[vlf_mask])
    vlf_power_ratio = vlf_power / (total_power + 1e-10)

    # Get psdpar.m parameters (10 additional features)
    psd_params = compute_psd_parameters(f, Pxx)

    return {
        'spectral_entropy': spectral_entropy,
        'peak_freq': peak_freq,
        'cardiac_power_ratio': cardiac_power_ratio,
        'high_freq_power_ratio': high_freq_power_ratio,
        'vlf_power_ratio': vlf_power_ratio,
        **psd_params  # 10 new parameters from psdpar.m
    }




# =============================================================================
# STATISTICAL FEATURES (Li & Clifford)
# =============================================================================

def extract_statistical_features(signal: np.ndarray) -> dict:
    """
    Extract statistical (higher moment) features.

    Based on Li & Clifford approach - distribution characteristics
    help identify artifacts.

    Args:
        signal: ECG signal (1D)

    Returns:
        Dictionary with statistical features:
        - mean, std, range
        - skewness: Asymmetry of distribution
        - kurtosis: Tailedness of distribution
        - zero_crossings: Rate of zero crossings

    Theory:
        - Normal ECG has moderate skewness/kurtosis
        - Artifacts cause extreme values
        - High zero-crossing rate indicates noise
    """
    signal = np.asarray(signal).flatten()

    # Basic statistics
    sig_mean = np.mean(signal)
    sig_std = np.std(signal)
    sig_range = np.max(signal) - np.min(signal)

    # Note: Higher moments (skewness/kurtosis) on raw signal removed
    # - They caused overflow (10^17) with artifact spikes
    # - Spectral versions from psdpar.m are used in classifier instead

    # Zero crossings (on mean-centered signal)
    centered = signal - sig_mean
    zero_crossings = np.sum(np.diff(np.sign(centered)) != 0)
    zero_crossing_rate = zero_crossings / len(signal)

    # Amplitude statistics
    abs_signal = np.abs(signal - sig_mean)
    max_amplitude = np.max(abs_signal)
    mean_amplitude = np.mean(abs_signal)

    return {
        'mean': sig_mean,
        'std': sig_std,
        'range': sig_range,
        'zero_crossing_rate': zero_crossing_rate,
        'max_amplitude': max_amplitude,
        'mean_amplitude': mean_amplitude
    }


# =============================================================================
# TIME-FREQUENCY FEATURES (BSA Course Requirement)
# =============================================================================

def extract_timefreq_features(signal: np.ndarray, fs: float) -> dict:
    """
    Extract time-frequency (STFT) features.

    BSA Course Requirement: Use spectrogram for non-stationary analysis.

    Args:
        signal: ECG signal (1D)
        fs: Sampling frequency (Hz)

    Returns:
        Dictionary with TF features:
        - temporal_variance: Variance across time frames
        - spectral_flatness: Flatness of spectrum over time
        - artifact_frame_ratio: Ratio of frames with high power transients
        - stationarity_index: How stationary the signal is

    Theory:
        - Good ECG: Quasi-periodic, low temporal variance
        - Bad ECG: Artifacts cause sudden power changes
        - STFT reveals time-varying spectral content
    """
    signal = np.asarray(signal).flatten()

    # STFT parameters (from BSA course)
    nperseg = int(fs * 0.5)  # 500ms window
    noverlap = int(nperseg * 0.75)  # 75% overlap

    # Compute spectrogram
    f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg,
                            noverlap=noverlap, window='hamming')

    # Focus on cardiac frequency range (0.5-30 Hz)
    cardiac_mask = (f >= 0.5) & (f <= 30)
    Sxx_cardiac = Sxx[cardiac_mask, :]

    # Temporal variance: How much power changes over time
    frame_power = np.sum(Sxx_cardiac, axis=0)
    temporal_variance = np.var(frame_power) / (np.mean(frame_power) + 1e-10)

    # Time-frequency spectral flatness (geometric mean / arithmetic mean)
    # Note: Named tf_spectral_flatness to distinguish from spectral_flatness in spectral_params.py
    # This operates on cardiac band (0.5-30Hz) STFT, while spectral_flatness uses full PSD
    eps = 1e-10
    flatness_per_frame = []
    for i in range(Sxx_cardiac.shape[1]):
        frame = Sxx_cardiac[:, i] + eps
        geo_mean = np.exp(np.mean(np.log(frame)))
        arith_mean = np.mean(frame)
        flatness_per_frame.append(geo_mean / arith_mean)
    tf_spectral_flatness = np.mean(flatness_per_frame)

    # Artifact detection: Frames with power > 3 std above mean
    power_threshold = np.mean(frame_power) + 3 * np.std(frame_power)
    artifact_frames = np.sum(frame_power > power_threshold)
    artifact_frame_ratio = artifact_frames / len(frame_power)

    # Stationarity index: Low variance = stationary
    stationarity_index = 1.0 / (1.0 + temporal_variance)

    return {
        'temporal_variance': temporal_variance,
        'tf_spectral_flatness': tf_spectral_flatness,
        'artifact_frame_ratio': artifact_frame_ratio,
        'stationarity_index': stationarity_index,
        'n_frames': len(t)
    }


# =============================================================================
# LEAD CORRELATION FEATURES (Maan et al.)
# =============================================================================

def extract_lead_correlation(signals: np.ndarray) -> dict:
    """
    Extract inter-lead correlation features.

    Based on Maan et al. - leads should be correlated in good ECG.

    Args:
        signals: Multi-lead ECG (samples, leads)

    Returns:
        Dictionary with correlation features:
        - mean_correlation: Mean pairwise correlation
        - min_correlation: Minimum pairwise correlation
        - correlation_std: Standard deviation of correlations

    Theory:
        - Good ECG: Leads are correlated (same cardiac source)
        - Bad ECG: Low correlation (noise is uncorrelated)
    """
    if signals.ndim == 1:
        return {
            'mean_correlation': 1.0,
            'min_correlation': 1.0,
            'correlation_std': 0.0
        }

    n_leads = signals.shape[1]
    correlations = []

    for i in range(n_leads):
        for j in range(i + 1, n_leads):
            # Skip leads with zero variance to avoid division by zero warning
            if np.std(signals[:, i]) < 1e-10 or np.std(signals[:, j]) < 1e-10:
                continue
            corr = np.corrcoef(signals[:, i], signals[:, j])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    if len(correlations) == 0:
        return {
            'mean_correlation': 0.0,
            'min_correlation': 0.0,
            'correlation_std': 0.0
        }

    return {
        'mean_correlation': np.mean(correlations),
        'min_correlation': np.min(correlations),
        'correlation_std': np.std(correlations)
    }


# =============================================================================
# COMBINED FEATURE EXTRACTION
# =============================================================================

@dataclass
class ECGFeatures:
    """Container for all ECG features."""
    snr: dict
    baseline: dict
    morphological: dict
    spectral: dict
    statistical: dict
    timefreq: dict
    lead_correlation: dict

    def to_dict(self) -> dict:
        """Flatten all features into a single dictionary."""
        result = {}
        for name, features in [
            ('snr', self.snr),
            ('baseline', self.baseline),
            ('morph', self.morphological),
            ('spectral', self.spectral),
            ('stat', self.statistical),
            ('tf', self.timefreq),
            ('lead', self.lead_correlation)
        ]:
            for key, value in features.items():
                result[f'{name}_{key}'] = value
        return result


def extract_all_features(
    signal: np.ndarray,
    fs: float,
    r_peaks: np.ndarray | None = None
) -> ECGFeatures:
    """
    Extract all features from an ECG signal.

    Args:
        signal: ECG signal (1D for single lead, 2D for multi-lead)
        fs: Sampling frequency (Hz)
        r_peaks: R-peak indices (optional, will detect from lead II or first lead)

    Returns:
        ECGFeatures dataclass with all feature categories
    """
    # Handle multi-lead vs single-lead
    if signal.ndim == 2:
        # Use lead II (index 1) for single-lead features, or first lead
        lead_idx = 1 if signal.shape[1] > 1 else 0
        single_lead = signal[:, lead_idx]
        multi_lead = signal
    else:
        single_lead = signal.flatten()
        multi_lead = signal.reshape(-1, 1)

    # Detect R-peaks if not provided
    if r_peaks is None:
        r_peaks = detect_r_peaks(single_lead, fs)

    # Extract all feature categories
    snr_features = extract_snr_features(single_lead, fs, r_peaks)
    baseline_features = extract_baseline_features(single_lead, fs)
    morph_features = extract_morphological_features(single_lead, fs, r_peaks)
    spectral_features = extract_spectral_features(single_lead, fs)
    stat_features = extract_statistical_features(single_lead)
    tf_features = extract_timefreq_features(single_lead, fs)
    lead_features = extract_lead_correlation(multi_lead)

    return ECGFeatures(
        snr=snr_features,
        baseline=baseline_features,
        morphological=morph_features,
        spectral=spectral_features,
        statistical=stat_features,
        timefreq=tf_features,
        lead_correlation=lead_features
    )


def extract_features_batch(
    dataset,
    max_records: int | None = None,
    verbose: bool = True
) -> tuple:
    """
    Extract features from entire dataset.

    Args:
        dataset: ECGDataset instance
        max_records: Maximum records to process (None for all)
        verbose: Print progress

    Returns:
        (features_list, labels_list) where each is a list
    """
    import pandas as pd

    features_list = []
    labels_list = []

    n_records = min(len(dataset), max_records) if max_records else len(dataset)

    for i in range(n_records):
        try:
            ecg = dataset[i]
            features = extract_all_features(ecg.signal, ecg.fs)
            features_dict = features.to_dict()
            features_dict['record_id'] = ecg.record_id

            features_list.append(features_dict)
            labels_list.append(1 if ecg.quality == 'Acceptable' else 0)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_records} records...")

        except Exception as e:
            if verbose:
                print(f"  Error processing record {i}: {e}")
            continue

    return features_list, labels_list
