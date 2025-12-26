"""
Pan-Tompkins QRS Detection Algorithm (Enhanced)

Python implementation based on:
- PSB_master/pan_tompkin.m (Hooman Sedghamiz, Linkoping University)
- Pan & Tompkins, IEEE Trans. Biomed. Eng., 1985

Enhanced with techniques from:
- WFDB XQRS: Ricker wavelet, learning phase, backsearch
- sleepecg: Flat data detection

Pipeline:
    Raw ECG → Bandpass (5-15 Hz) → Derivative/Wavelet → Squaring → Moving Average → Adaptive Threshold
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from dataclasses import dataclass


def ricker(points: int, a: float) -> np.ndarray:
    """
    Ricker wavelet (Mexican hat wavelet).

    Reimplemented since scipy removed it in newer versions.
    The Ricker wavelet is the normalized second derivative of a Gaussian.

    Args:
        points: Number of points in the wavelet
        a: Width parameter (controls frequency response)

    Returns:
        Ricker wavelet array

    Formula:
        ψ(t) = (2 / (√3σπ^(1/4))) * (1 - (t/σ)²) * exp(-(t/σ)²/2)
        where σ = a

    Reference:
        Ryan, 1994, "Ricker, Ormsby, Klauder, Butterworth—A Choice of Wavelets"
    """
    # Create symmetric time vector centered at 0
    t = np.arange(0, points) - (points - 1) / 2

    # Normalize by width parameter
    t_norm = t / a

    # Ricker wavelet formula
    A = 2 / (np.sqrt(3 * a) * (np.pi ** 0.25))
    wavelet = A * (1 - t_norm ** 2) * np.exp(-t_norm ** 2 / 2)

    return wavelet


@dataclass
class PanTompkinsResult:
    """Results from Pan-Tompkins QRS detection."""
    qrs_peaks: np.ndarray      # R-peak indices in original signal
    qrs_amplitudes: np.ndarray # R-peak amplitudes
    filtered_signal: np.ndarray # Bandpass filtered signal
    mwi_signal: np.ndarray     # Moving window integrator output
    delay: int                 # Total delay in samples


def pan_tompkins(ecg: np.ndarray, fs: float, invert: bool | None = None) -> PanTompkinsResult:
    """
    Pan-Tompkins QRS detection algorithm.

    Args:
        ecg: Raw ECG signal (1D array)
        fs: Sampling frequency (Hz)
        invert: If True, invert signal before processing.
                If None (default), auto-detect polarity.

    Returns:
        PanTompkinsResult with detected QRS peaks and intermediate signals

    Reference:
        Pan & Tompkins, "A Real-Time QRS Detection Algorithm",
        IEEE Trans. Biomed. Eng., vol. BME-32, no. 3, March 1985.
    """
    ecg = np.asarray(ecg).flatten()
    N = len(ecg)

    # Auto-detect polarity: if signal is predominantly negative, invert
    if invert is None:
        # Check if negative peaks are larger than positive peaks
        invert = np.abs(np.min(ecg)) > np.abs(np.max(ecg))

    if invert:
        ecg = -ecg
    delay = 0

    # ==========================================
    # STEP 1: Bandpass Filter (5-15 Hz)
    # ==========================================
    # Removes baseline wander (<5 Hz) and high-frequency noise (>15 Hz)
    # Highlights QRS complex which has most energy in 5-15 Hz range

    if fs == 200:
        # Original Pan-Tompkins filter design for 200 Hz
        # Low-pass: H(z) = ((1 - z^(-6))^2)/(1 - z^(-1))^2
        b_lp = [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
        a_lp = [1, -2, 1]
        ecg_l = np.convolve(ecg, np.array(b_lp) / np.sum(np.abs(b_lp)), mode='same')
        delay += 6

        # High-pass: H(z) = (-1+32z^(-16)+z^(-32))/(1+z^(-1))
        b_hp = np.zeros(33)
        b_hp[0] = -1
        b_hp[16] = 32
        b_hp[17] = -32
        b_hp[32] = 1
        a_hp = [1, -1]
        ecg_h = np.convolve(ecg_l, b_hp / np.sum(np.abs(b_hp)), mode='same')
        delay += 16
    else:
        # Butterworth bandpass for other sampling frequencies
        f1, f2 = 5, 15  # Cutoff frequencies
        nyq = fs / 2
        Wn = [f1 / nyq, f2 / nyq]
        b, a = butter(3, Wn, btype='band')
        ecg_h = filtfilt(b, a, ecg)

    # Normalize
    ecg_h = ecg_h / (np.max(np.abs(ecg_h)) + 1e-10)

    # ==========================================
    # STEP 2: Derivative Filter
    # ==========================================
    # H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^(2))
    # Highlights the steep slopes of QRS complex

    h_d = np.array([-1, -2, 0, 2, 1]) / 8
    ecg_d = np.convolve(ecg_h, h_d, mode='same')
    ecg_d = ecg_d / (np.max(np.abs(ecg_d)) + 1e-10)
    delay += 2

    # ==========================================
    # STEP 3: Squaring
    # ==========================================
    # Makes all values positive and emphasizes large differences
    # Non-linearly amplifies high-frequency components (QRS)

    ecg_s = ecg_d ** 2

    # ==========================================
    # STEP 4: Moving Window Integration
    # ==========================================
    # Window width = 150ms (recommended by Pan-Tompkins)
    # Smooths the signal and provides waveform feature information

    window_width = int(0.150 * fs)
    mwi_kernel = np.ones(window_width) / window_width
    ecg_m = np.convolve(ecg_s, mwi_kernel, mode='same')
    delay += window_width // 2

    # ==========================================
    # STEP 5: Adaptive Thresholding
    # ==========================================
    # Find peaks with minimum distance (refractory period = 200ms)

    min_distance = int(0.2 * fs)  # 200ms refractory period
    peaks, properties = find_peaks(ecg_m, distance=min_distance)

    if len(peaks) == 0:
        return PanTompkinsResult(
            qrs_peaks=np.array([]),
            qrs_amplitudes=np.array([]),
            filtered_signal=ecg_h,
            mwi_signal=ecg_m,
            delay=delay
        )

    # Initialize thresholds (2-second training phase)
    training_samples = min(int(2 * fs), N)
    THR_SIG = np.max(ecg_m[:training_samples]) / 3
    THR_NOISE = np.mean(ecg_m[:training_samples]) / 2
    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE

    qrs_peaks = []
    qrs_amplitudes = []

    for i, peak in enumerate(peaks):
        peak_val = ecg_m[peak]

        if peak_val > THR_SIG:
            # Signal peak detected
            qrs_peaks.append(peak)
            qrs_amplitudes.append(ecg_h[peak])

            # Update signal level
            SIG_LEV = 0.125 * peak_val + 0.875 * SIG_LEV

        elif peak_val > THR_NOISE:
            # Noise peak
            NOISE_LEV = 0.125 * peak_val + 0.875 * NOISE_LEV

        # Update thresholds
        THR_SIG = NOISE_LEV + 0.25 * (SIG_LEV - NOISE_LEV)
        THR_NOISE = 0.5 * THR_SIG

    qrs_peaks = np.array(qrs_peaks)
    qrs_amplitudes = np.array(qrs_amplitudes)

    # ==========================================
    # STEP 6: Refine peak locations
    # ==========================================
    # Find the actual R-peak in the bandpass filtered signal

    refined_peaks = []
    search_window = int(0.150 * fs)

    for peak in qrs_peaks:
        start = max(0, peak - search_window)
        end = min(N, peak + search_window // 2)

        # Find max in filtered signal
        local_max_idx = np.argmax(np.abs(ecg_h[start:end]))
        refined_peaks.append(start + local_max_idx)

    refined_peaks = np.array(refined_peaks)

    return PanTompkinsResult(
        qrs_peaks=refined_peaks,
        qrs_amplitudes=ecg_h[refined_peaks] if len(refined_peaks) > 0 else np.array([]),
        filtered_signal=ecg_h,
        mwi_signal=ecg_m,
        delay=delay
    )


def detect_qrs_pan_tompkins(signal: np.ndarray, fs: float, enhanced: bool = True) -> np.ndarray:
    """
    Simplified interface - returns only QRS peak indices.

    Args:
        signal: ECG signal (1D)
        fs: Sampling frequency (Hz)
        enhanced: Use enhanced algorithm with XQRS improvements (default: True)

    Returns:
        Array of R-peak indices
    """
    if enhanced:
        result = pan_tompkins_enhanced(signal, fs)
    else:
        result = pan_tompkins(signal, fs)
    return result.qrs_peaks


# ============================================================================
# ENHANCED PAN-TOMPKINS (with XQRS/sleepecg improvements)
# ============================================================================

def detect_flat_signal(ecg: np.ndarray, fs: float, threshold: float = 1e-6) -> int:
    """
    Detect flat (zero) signal at the start of recording.

    Based on sleepecg implementation: skip initial flat data that would
    corrupt threshold initialization.

    Args:
        ecg: ECG signal
        fs: Sampling frequency
        threshold: Variance threshold for flat detection

    Returns:
        Index where non-flat data begins
    """
    window_size = int(0.1 * fs)  # 100ms windows

    for i in range(0, len(ecg) - window_size, window_size):
        window = ecg[i:i + window_size]
        if np.var(window) > threshold:
            return max(0, i - window_size)  # Go back one window for safety

    return 0


def ricker_wavelet_filter(ecg: np.ndarray, fs: float, qrs_width: float = 0.1) -> np.ndarray:
    """
    Apply Ricker (Mexican hat) wavelet filter for QRS enhancement.

    Based on WFDB XQRS: The Ricker wavelet is better matched to QRS
    morphology than a simple derivative filter.

    Args:
        ecg: Bandpass filtered ECG
        fs: Sampling frequency
        qrs_width: Expected QRS width in seconds (default: 0.1s)

    Returns:
        Wavelet-filtered signal

    Theory:
        The Ricker wavelet (2nd derivative of Gaussian) has a shape
        similar to QRS complexes, making it an optimal matched filter.
    """
    # Wavelet points should span about one QRS width
    points = int(qrs_width * fs)
    if points < 4:
        points = 4
    if points % 2 == 0:
        points += 1  # Ensure odd length for symmetry

    # Width parameter (a) controls the wavelet's frequency response
    # a=4 gives good QRS detection as used in XQRS
    wavelet = ricker(points, 4)

    # Convolve (correlation for matched filtering)
    filtered = np.convolve(ecg, wavelet, mode='same')

    return filtered


def pan_tompkins_enhanced(
    ecg: np.ndarray,
    fs: float,
    invert: bool | None = None,
    use_wavelet: bool = True
) -> PanTompkinsResult:
    """
    Enhanced Pan-Tompkins with XQRS/sleepecg improvements.

    Improvements over original:
    1. Flat signal detection (sleepecg)
    2. Ricker wavelet filter option (XQRS)
    3. Learning phase for threshold calibration (XQRS)
    4. Backsearch mechanism for missed beats (XQRS)

    Args:
        ecg: Raw ECG signal (1D array)
        fs: Sampling frequency (Hz)
        invert: If True, invert signal. If None, auto-detect.
        use_wavelet: Use Ricker wavelet instead of derivative (default: True)

    Returns:
        PanTompkinsResult with detected QRS peaks
    """
    ecg = np.asarray(ecg).flatten()
    N = len(ecg)

    # ==========================================
    # IMPROVEMENT 1: Flat signal detection
    # ==========================================
    flat_end = detect_flat_signal(ecg, fs)
    if flat_end > 0:
        # Process only non-flat portion, then adjust indices
        ecg_work = ecg[flat_end:]
    else:
        ecg_work = ecg
        flat_end = 0

    N_work = len(ecg_work)
    if N_work < int(2 * fs):  # Need at least 2 seconds
        return PanTompkinsResult(
            qrs_peaks=np.array([]),
            qrs_amplitudes=np.array([]),
            filtered_signal=ecg,
            mwi_signal=np.zeros_like(ecg),
            delay=0
        )

    # Auto-detect polarity
    if invert is None:
        invert = np.abs(np.min(ecg_work)) > np.abs(np.max(ecg_work))

    if invert:
        ecg_work = -ecg_work

    delay = 0

    # ==========================================
    # STEP 1: Bandpass Filter (5-15 Hz)
    # ==========================================
    f1, f2 = 5, 15
    nyq = fs / 2
    Wn = [f1 / nyq, f2 / nyq]
    b, a = butter(3, Wn, btype='band')
    ecg_h = filtfilt(b, a, ecg_work)
    ecg_h = ecg_h / (np.max(np.abs(ecg_h)) + 1e-10)

    # ==========================================
    # STEP 2: Derivative or Wavelet Filter
    # ==========================================
    if use_wavelet:
        # IMPROVEMENT 2: Ricker wavelet (XQRS approach)
        ecg_d = ricker_wavelet_filter(ecg_h, fs)
    else:
        # Original derivative filter
        h_d = np.array([-1, -2, 0, 2, 1]) / 8
        ecg_d = np.convolve(ecg_h, h_d, mode='same')

    ecg_d = ecg_d / (np.max(np.abs(ecg_d)) + 1e-10)
    delay += 2

    # ==========================================
    # STEP 3: Squaring
    # ==========================================
    ecg_s = ecg_d ** 2

    # ==========================================
    # STEP 4: Moving Window Integration
    # ==========================================
    window_width = int(0.150 * fs)
    mwi_kernel = np.ones(window_width) / window_width
    ecg_m = np.convolve(ecg_s, mwi_kernel, mode='same')
    delay += window_width // 2

    # ==========================================
    # STEP 5: Peak Detection with Adaptive Thresholding
    # ==========================================
    min_distance = int(0.2 * fs)  # 200ms refractory period
    peaks, _ = find_peaks(ecg_m, distance=min_distance)

    if len(peaks) == 0:
        # Pad filtered signal back to original size
        ecg_h_full = np.zeros(N)
        ecg_h_full[flat_end:flat_end + N_work] = ecg_h if not invert else -ecg_h
        ecg_m_full = np.zeros(N)
        ecg_m_full[flat_end:flat_end + N_work] = ecg_m

        return PanTompkinsResult(
            qrs_peaks=np.array([]),
            qrs_amplitudes=np.array([]),
            filtered_signal=ecg_h_full,
            mwi_signal=ecg_m_full,
            delay=delay
        )

    # ==========================================
    # IMPROVEMENT 3: Learning Phase (XQRS)
    # ==========================================
    # Use first 2 seconds to calibrate thresholds more robustly
    learning_samples = min(int(2 * fs), N_work)
    learning_peaks = peaks[peaks < learning_samples]

    if len(learning_peaks) >= 2:
        # Use learning phase peaks to initialize thresholds
        learning_amplitudes = ecg_m[learning_peaks]
        THR_SIG = np.mean(learning_amplitudes) * 0.5
        THR_NOISE = THR_SIG * 0.5
        SIG_LEV = np.mean(learning_amplitudes)
        NOISE_LEV = THR_NOISE
    else:
        # Fallback to original initialization
        THR_SIG = np.max(ecg_m[:learning_samples]) / 3
        THR_NOISE = np.mean(ecg_m[:learning_samples]) / 2
        SIG_LEV = THR_SIG
        NOISE_LEV = THR_NOISE

    qrs_peaks = []
    qrs_amplitudes = []
    last_qrs_idx = 0
    rr_intervals = []

    for i, peak in enumerate(peaks):
        peak_val = ecg_m[peak]

        if peak_val > THR_SIG:
            # Signal peak detected
            qrs_peaks.append(peak)
            qrs_amplitudes.append(ecg_h[peak])

            # Track RR intervals for backsearch
            if last_qrs_idx > 0:
                rr_intervals.append(peak - last_qrs_idx)
            last_qrs_idx = peak

            # Update signal level
            SIG_LEV = 0.125 * peak_val + 0.875 * SIG_LEV

        elif peak_val > THR_NOISE:
            # Noise peak
            NOISE_LEV = 0.125 * peak_val + 0.875 * NOISE_LEV

        # Update thresholds
        THR_SIG = NOISE_LEV + 0.25 * (SIG_LEV - NOISE_LEV)
        THR_NOISE = 0.5 * THR_SIG

        # ==========================================
        # IMPROVEMENT 4: Backsearch Mechanism (XQRS)
        # ==========================================
        # If no QRS detected for 1.66 × mean RR, search back with lower threshold
        if len(rr_intervals) >= 2 and len(qrs_peaks) > 0:
            mean_rr = np.mean(rr_intervals[-8:])  # Use recent RR intervals
            samples_since_last = peak - last_qrs_idx

            if samples_since_last > 1.66 * mean_rr:
                # Missed beat likely - search back with halved threshold
                THR_BACK = 0.5 * THR_SIG
                search_start = last_qrs_idx + int(0.2 * fs)  # After refractory
                search_end = peak

                # Find peaks in the gap that exceed backsearch threshold
                back_peaks = [p for p in peaks if search_start < p < search_end]
                for bp in back_peaks:
                    if ecg_m[bp] > THR_BACK and bp not in qrs_peaks:
                        # Insert missed beat
                        qrs_peaks.append(bp)
                        qrs_amplitudes.append(ecg_h[bp])
                        rr_intervals.append(bp - last_qrs_idx)
                        last_qrs_idx = bp

    # Sort peaks (backsearch may have added out of order)
    if len(qrs_peaks) > 0:
        sort_idx = np.argsort(qrs_peaks)
        qrs_peaks = np.array(qrs_peaks)[sort_idx]
        qrs_amplitudes = np.array(qrs_amplitudes)[sort_idx]
    else:
        qrs_peaks = np.array([])
        qrs_amplitudes = np.array([])

    # ==========================================
    # STEP 6: Refine peak locations
    # ==========================================
    refined_peaks = []
    search_window = int(0.150 * fs)

    for peak in qrs_peaks:
        start = max(0, peak - search_window)
        end = min(N_work, peak + search_window // 2)
        local_max_idx = np.argmax(np.abs(ecg_h[start:end]))
        refined_peaks.append(start + local_max_idx)

    refined_peaks = np.array(refined_peaks)

    # Adjust indices for flat signal offset
    if flat_end > 0:
        refined_peaks = refined_peaks + flat_end

    # Pad signals back to original size
    ecg_h_full = np.zeros(N)
    ecg_h_full[flat_end:flat_end + N_work] = ecg_h if not invert else -ecg_h
    ecg_m_full = np.zeros(N)
    ecg_m_full[flat_end:flat_end + N_work] = ecg_m

    return PanTompkinsResult(
        qrs_peaks=refined_peaks,
        qrs_amplitudes=ecg_h_full[refined_peaks] if len(refined_peaks) > 0 else np.array([]),
        filtered_signal=ecg_h_full,
        mwi_signal=ecg_m_full,
        delay=delay
    )
