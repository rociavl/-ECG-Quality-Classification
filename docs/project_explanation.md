# ECG Quality Assessment - Complete Project Explanation

**Course:** Biomedical Signal Analysis (BSA)
**Project:** PhysioNet/CinC Challenge 2011 - ECG Quality Classification

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Phase 1: Project Setup](#3-phase-1-project-setup)
4. [Phase 2: Data Loading](#4-phase-2-data-loading)
5. [Phase 3: Preprocessing](#5-phase-3-preprocessing)
   - 5.1 [Filtering Theory](#51-filtering-theory)
   - 5.2 [Pan-Tompkins QRS Detection](#52-pan-tompkins-qrs-detection)
   - 5.3 [Cardiac Frequency Estimation](#53-cardiac-frequency-estimation)
6. [Phase 4: Feature Extraction](#6-phase-4-feature-extraction)
7. [Phase 5: Classification](#7-phase-5-classification)
8. [Phase 6: Spectral Parameters (psdpar.m)](#8-phase-6-spectral-parameters-psdparm)
9. [Phase 7: Results](#9-phase-7-results)
10. [Generated Figures](#10-generated-figures)
11. [References](#11-references)

---

## 1. Project Overview

### Goal
Classify 12-lead ECG recordings as **Acceptable** or **Unacceptable** for clinical interpretation, targeting mobile health (mHealth) applications where signal quality varies significantly.

### Why This Matters
- Mobile ECG devices are increasingly used for remote patient monitoring
- Poor quality ECGs can lead to misdiagnosis
- Automated quality assessment enables real-time feedback to users
- Challenge goal: "Would you trust a diagnosis from this ECG?"

### Our Approach
```
Raw ECG → Preprocessing → Feature Extraction → Classification → Quality Label
```

---

## 2. Dataset Description

### PhysioNet/CinC Challenge 2011

| Specification | Value |
|--------------|-------|
| Database | PhysioNet Challenge 2011 (set-a) |
| Total records | 1,000 (training set) |
| Duration | 10 seconds per record |
| Leads | 12-lead standard ECG |
| Sampling rate | 500 Hz |
| Resolution | 16-bit ADC |
| Amplitude resolution | 5 µV/LSB |
| Bandwidth | 0.05 - 100 Hz |
| Classes | Acceptable (~77%), Unacceptable (~23%) |

### Standard 12-Lead Configuration
```
Limb Leads:        I, II, III (bipolar)
Augmented Leads:   aVR, aVL, aVF (unipolar)
Precordial Leads:  V1, V2, V3, V4, V5, V6 (chest)
```

### Why Lead II?
We primarily analyze **Lead II** because:
- Best visibility of P-wave (atrial depolarization)
- Clear QRS complex (ventricular depolarization)
- Standard lead for rhythm analysis
- Aligned with cardiac electrical axis

---

## 3. Phase 1: Project Setup

### Directory Structure
```
Signals_Project/
├── src/                    # Python modules
│   ├── __init__.py
│   ├── data_loader.py      # ECG loading utilities
│   ├── filters.py          # Signal filtering functions
│   ├── pan_tompkins.py     # QRS detection algorithm
│   ├── qrs_detection.py    # R-peak detection wrapper
│   ├── cardiac_frequency.py # HR estimation (Fourier/Burg)
│   ├── preprocessing.py    # Combined pipeline
│   └── features.py         # Feature extraction
├── scripts/
│   ├── download_dataset.py # PhysioNet downloader
│   └── generate_figures.py # Presentation figures
├── data/challenge2011/set-a/  # ECG records
├── results/figures/        # Output plots
└── docs/                   # Documentation
```

### Dependencies
- **numpy/scipy**: Numerical computing and signal processing
- **wfdb**: PhysioNet data format reading
- **neurokit2**: ECG processing library (for comparison)
- **matplotlib**: Visualization

---

## 4. Phase 2: Data Loading

### Implementation: `src/data_loader.py`

```python
class ECGRecord:
    record_id: str          # e.g., "1002867"
    signal: np.ndarray      # Shape: (5000, 12) = 10s × 12 leads
    fs: int                 # 500 Hz
    leads: list[str]        # ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1'-'V6']
    quality: str            # 'Acceptable' or 'Unacceptable'

class ECGDataset:
    # Provides indexing, iteration, and filtering by quality
    dataset.acceptable      # List of acceptable record IDs
    dataset.unacceptable    # List of unacceptable record IDs
```

### Data Format
PhysioNet uses WFDB format:
- `.hea` - Header file (metadata)
- `.dat` - Binary signal data
- Labels from `RECORDS-acceptable` and `RECORDS-unacceptable` files

---

## 5. Phase 3: Preprocessing

### 5.1 Filtering Theory

#### Why Filter ECG Signals?

ECG signals contain:
1. **Useful information** (0.5 - 40 Hz): P-wave, QRS complex, T-wave
2. **Noise sources**:
   - Baseline wander (< 0.5 Hz): Respiration, movement
   - Powerline interference (50/60 Hz): Electrical mains
   - Muscle noise (> 100 Hz): EMG contamination
   - Electrode motion artifacts: Sudden spikes

#### AHA Recommended Bandwidth

| Application | Low Cutoff | High Cutoff |
|-------------|------------|-------------|
| Diagnostic ECG | 0.05 Hz | 150 Hz |
| Monitoring ECG | 0.5 Hz | 40 Hz |
| **Our Project** | **0.5 Hz** | **100 Hz** |

#### Implementation: `src/filters.py`

**Bandpass Filter (Butterworth)**
```python
def bandpass_filter(signal, fs, lowcut=0.5, highcut=100, order=4):
    nyq = fs / 2
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)  # Zero-phase filtering
```

**Theory - Butterworth Filter:**
- Maximally flat frequency response in passband
- No ripples (unlike Chebyshev)
- Order N determines roll-off steepness: -20N dB/decade

**Zero-Phase Filtering (filtfilt):**
- Applies filter forward, then backward
- Eliminates phase distortion
- Doubles the filter order effectively
- Critical for preserving ECG morphology

**Notch Filter (50 Hz removal)**
```python
def notch_filter(signal, fs, freq=50, Q=30):
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal)
```

**Theory - Quality Factor Q:**
- Q = f_center / bandwidth
- Higher Q = narrower notch
- Q = 30 gives ~1.7 Hz bandwidth at 50 Hz

#### Filtering Pipeline Visualization
```
Original ECG
     ↓
High-pass (0.5 Hz) → Removes baseline wander
     ↓
Notch (50 Hz) → Removes powerline interference
     ↓
Low-pass (100 Hz) → Removes high-frequency noise
     ↓
Clean ECG for analysis
```

---

### 5.2 Pan-Tompkins QRS Detection

#### Algorithm Overview

The Pan-Tompkins algorithm (1985) is the gold standard for real-time QRS detection. It uses a series of filters to enhance QRS complexes while suppressing noise.

**Reference:** Pan & Tompkins, "A Real-Time QRS Detection Algorithm", IEEE Trans. Biomed. Eng., 1985

#### Implementation: `src/pan_tompkins.py`

#### Step-by-Step Pipeline

```
Raw ECG → Bandpass → Derivative → Squaring → Integration → Thresholding → R-peaks
```

**Step 1: Bandpass Filter (5-15 Hz)**
```python
# Butterworth bandpass
b, a = butter(3, [5/nyq, 15/nyq], btype='band')
filtered = filtfilt(b, a, signal)
```

*Theory:* QRS complex has most energy in 5-15 Hz range. This filter:
- Removes baseline wander (< 5 Hz)
- Removes high-frequency noise (> 15 Hz)
- Preserves QRS slope information

**Step 2: Derivative Filter**
```python
# 5-point derivative: H(z) = (1/8)(-z^-2 - 2z^-1 + 2z + z^2)
h_d = np.array([-1, -2, 0, 2, 1]) / 8
derivative = np.convolve(filtered, h_d, mode='same')
```

*Theory:* The derivative highlights rapid changes (steep slopes) in the signal. QRS complexes have the steepest slopes in the ECG.

**Step 3: Squaring**
```python
squared = derivative ** 2
```

*Theory:* Squaring serves two purposes:
1. Makes all values positive
2. Nonlinearly amplifies large values (QRS) relative to small values (noise)

**Step 4: Moving Window Integration**
```python
# Window width = 150 ms (recommended by Pan-Tompkins)
window_width = int(0.150 * fs)
mwi = np.convolve(squared, np.ones(window_width)/window_width, mode='same')
```

*Theory:* Integration smooths multiple peaks in QRS into a single pulse. Window width of 150ms captures the typical QRS duration.

**Step 5: Adaptive Thresholding**
```python
# Dual threshold system
THR_SIG = NOISE_LEV + 0.25 * (SIG_LEV - NOISE_LEV)
THR_NOISE = 0.5 * THR_SIG

# Update thresholds based on detected peaks
if peak > THR_SIG:
    SIG_LEV = 0.125 * peak + 0.875 * SIG_LEV    # Signal peak
else:
    NOISE_LEV = 0.125 * peak + 0.875 * NOISE_LEV # Noise peak
```

*Theory:* Adaptive thresholds adjust to:
- Different ECG amplitudes between patients
- Gradual amplitude changes within a recording
- The 0.125/0.875 weighting provides smooth adaptation

#### Additional Detection Rules

**Refractory Period (200 ms)**
```python
min_distance = int(0.2 * fs)  # Can't have two beats within 200ms
```
*Physiological basis:* The heart cannot depolarize twice within 200ms (absolute refractory period).

**Searchback Mechanism**
```python
if time_since_last_beat > 1.66 * mean_RR:
    # Missed beat - search back with lower threshold
    THR_BACK = 0.5 * THR_SIG
```
*Theory:* If no beat detected for 166% of the average RR interval, a beat was likely missed. Search backward with halved threshold.

#### Our Enhancements (from WFDB XQRS)

1. **Auto-polarity detection**: Handles inverted QRS complexes
2. **Ricker wavelet filter**: Better matched filter than simple derivative
3. **Learning phase**: First 2 seconds calibrate thresholds
4. **Flat signal detection**: Skips initial zero-signal regions

---

### 5.3 Cardiac Frequency Estimation

#### Implementation: `src/cardiac_frequency.py`

Based on BSA Lab 3 (Rocío Ávalos, Antonio Di Pierro).

#### Method 1: Fourier (Periodogram)

```python
def estimate_cardiac_frequency_fourier(signal, fs):
    f, Pxx = periodogram(signal, fs, window='hamming')

    # Find peak in cardiac range (0.5-2.5 Hz = 30-150 bpm)
    cardiac_mask = (f >= 0.5) & (f <= 2.5)
    peak_idx = np.argmax(Pxx[cardiac_mask])
    fcard = f[cardiac_mask][peak_idx]

    hr = fcard * 60  # Convert Hz to bpm
```

*Theory:* The periodogram estimates Power Spectral Density (PSD):
```
PSD(f) = |FFT(x)|² / N
```

The cardiac frequency appears as the fundamental peak in the spectrum. Harmonics (2×, 3× fundamental) may also be present due to QRS morphology.

#### Method 2: Burg (AR Model)

```python
def estimate_cardiac_frequency_burg(signal, fs, order=16):
    # Estimate AR coefficients using Burg method
    ar_coeffs = arburg(signal, order)

    # Find poles of AR model
    poles = np.roots(ar_coeffs)

    # Pole frequency = angle(pole) / pi * fs / 2
    frequencies = np.abs(np.angle(poles)) / np.pi * fs / 2
```

*Theory:* AR (Autoregressive) models represent the signal as:
```
x[n] = -Σ(a_k × x[n-k]) + e[n]
```

The Burg method estimates AR coefficients by minimizing forward and backward prediction errors. Spectral peaks correspond to poles close to the unit circle.

**Advantages of Burg method:**
- Smoother spectra than periodogram
- Better frequency resolution for short signals
- No spectral leakage

---

## 6. Phase 4: Feature Extraction

### Implementation: `src/features.py`

Features are organized by category, aligned with PhysioNet Challenge 2011 winning approaches.

### 6.1 SNR Features (Hayn et al.)

**Signal-to-Noise Ratio**
```python
def extract_snr_features(signal, fs, r_peaks):
    # QRS regions: 100ms window around each R-peak
    qrs_power = variance(signal[qrs_regions])
    noise_power = variance(signal[non_qrs_regions])

    snr_db = 10 * log10(qrs_power / noise_power)
```

*Theory:*
- **Good ECG:** SNR > 10 dB (QRS dominates)
- **Bad ECG:** SNR < 5 dB (noise dominates)

### 6.2 Baseline Features (Ho & Chen)

**Baseline Wander Detection**
```python
def extract_baseline_features(signal, fs):
    # Extract baseline using low-pass filter at 0.5 Hz
    baseline = lowpass_filter(signal, cutoff=0.5)

    baseline_power = np.var(baseline)
    baseline_range = np.max(baseline) - np.min(baseline)
```

*Theory:* Baseline wander is low-frequency drift caused by:
- Respiration (0.1-0.5 Hz)
- Body movement
- Poor electrode contact

High baseline power indicates poor quality.

### 6.3 Morphological Features (Kalkstein et al.)

**Beat-Based Analysis**
```python
def extract_morphological_features(signal, fs, r_peaks):
    beat_count = len(r_peaks)
    rr_intervals = np.diff(r_peaks) / fs  # in seconds

    heart_rate = 60 / np.mean(rr_intervals)  # bpm
    rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
```

*Theory - Expected Values for 10-second ECG:*

| Parameter | Normal Range | Meaning |
|-----------|--------------|---------|
| Beat count | 8-17 | 48-100 bpm |
| Heart rate | 60-100 bpm | Normal sinus rhythm |
| RR CV | < 0.2 | Regular rhythm |
| RR CV | > 0.3 | Irregular (artifact or arrhythmia) |

**Coefficient of Variation (CV):**
```
CV = σ_RR / μ_RR
```
- CV measures rhythm regularity independent of heart rate
- Low CV = regular rhythm
- High CV = irregular (possible artifact or arrhythmia)

### 6.4 Spectral Features (Xia et al.)

**Frequency Domain Analysis**
```python
def extract_spectral_features(signal, fs):
    f, Pxx = periodogram(signal, fs)

    # Spectral entropy (normalized)
    Pxx_norm = Pxx / sum(Pxx)
    spectral_entropy = -sum(Pxx_norm * log2(Pxx_norm)) / log2(len(Pxx))

    # Power distribution
    cardiac_power = sum(Pxx[0.5-40 Hz]) / total_power
    hf_power = sum(Pxx[>40 Hz]) / total_power
```

*Theory - Spectral Entropy:*
```
H = -Σ(p_i × log₂(p_i)) / log₂(N)
```
- Measures signal complexity/randomness
- Low entropy: Periodic signal (good ECG)
- High entropy: Noisy/random signal (bad ECG)

*Power Band Distribution:*
- **Cardiac band (0.5-40 Hz):** Should contain most ECG power
- **High frequency (>40 Hz):** Muscle noise, artifacts
- **Very low frequency (<0.5 Hz):** Baseline wander

### 6.5 Statistical Features (Li & Clifford)

**Higher-Order Statistics**
```python
def extract_statistical_features(signal):
    skewness = scipy.stats.skew(signal)
    kurtosis = scipy.stats.kurtosis(signal)
    zero_crossing_rate = count_zero_crossings(signal) / len(signal)
```

*Theory:*

**Skewness** - Distribution asymmetry:
```
γ₁ = E[(X-μ)³] / σ³
```
- Normal ECG: Moderate skewness
- Artifacts: Extreme skewness (one-sided spikes)

**Kurtosis** - Distribution "tailedness":
```
γ₂ = E[(X-μ)⁴] / σ⁴ - 3
```
- Normal ECG: Near-zero kurtosis
- Spiky artifacts: High kurtosis (heavy tails)

### 6.6 Time-Frequency Features (BSA Course Requirement)

**STFT Spectrogram Analysis**
```python
def extract_timefreq_features(signal, fs):
    # Short-Time Fourier Transform
    f, t, Sxx = spectrogram(signal, fs,
                            nperseg=int(0.5*fs),    # 500ms window
                            noverlap=int(0.375*fs)) # 75% overlap

    # Temporal variance - how power changes over time
    frame_power = sum(Sxx, axis=0)  # Power per time frame
    temporal_variance = var(frame_power) / mean(frame_power)

    # Stationarity index
    stationarity_index = 1 / (1 + temporal_variance)

    # Artifact detection
    threshold = mean(frame_power) + 3 * std(frame_power)
    artifact_frames = count(frame_power > threshold)
```

*Theory - Why Time-Frequency Analysis?*

ECG is **quasi-stationary**: spectral content changes over time (each heartbeat is slightly different). The spectrogram reveals:
- **Good ECG:** Regular pattern, consistent power across frames
- **Bad ECG:** Sudden power changes (artifacts), inconsistent patterns

**STFT Parameters:**
| Parameter | Value | Reason |
|-----------|-------|--------|
| Window | 500 ms | Captures 1-2 heartbeats |
| Overlap | 75% | Smooth time resolution |
| Frequency range | 0-30 Hz | Cardiac content |

**Trade-off (Heisenberg uncertainty):**
- Longer window → Better frequency resolution, worse time resolution
- Shorter window → Better time resolution, worse frequency resolution

### 6.7 Lead Correlation Features (Maan et al.)

**Inter-Lead Consistency**
```python
def extract_lead_correlation(signals):  # signals: (samples, 12 leads)
    correlations = []
    for i in range(12):
        for j in range(i+1, 12):
            corr = np.corrcoef(signals[:,i], signals[:,j])[0,1]
            correlations.append(abs(corr))

    mean_correlation = np.mean(correlations)
```

*Theory:* All 12 leads measure the same cardiac electrical activity from different angles. In a good ECG:
- Leads should be correlated (same source)
- Low correlation indicates noise (uncorrelated between leads)

---

## 7. Phase 5: Classification

### Implementation: `src/classifier.py`

A rule-based heuristic classifier following the Moody 2011 approach. Uses interpretable thresholds on extracted features.

### 7.1 Classifier Architecture

```python
class QualityClassifier:
    def __init__(self, thresholds: QualityThresholds):
        self.thresholds = thresholds

    def classify(self, signal, fs) -> ClassificationResult:
        features = extract_all_features(signal, fs)
        return self.classify_from_features(features)

    def evaluate(self, dataset) -> dict:
        # Returns accuracy, precision, recall, F1, confusion matrix
```

**Advantages:**
- **Interpretable:** Explains WHY each ECG was rejected
- **No training required:** Thresholds based on physiological reasoning
- **Fast execution:** Suitable for mobile/real-time applications

### 7.2 Classification Rules (16 Total)

| # | Rule | Threshold | Theory |
|---|------|-----------|--------|
| 1 | SNR | ≥ 5.0 dB | Signal vs noise power ratio |
| 2 | Beat count | 5-30 beats | Expected for 10s @ 30-180 bpm |
| 3 | Heart rate | 30-200 bpm | Physiological limits |
| 4 | RR regularity | CV ≤ 0.5 | Rhythm consistency |
| 5 | Baseline wander | ratio ≤ 0.3 | Low-frequency drift |
| 6 | Cardiac power | ≥ 50% | Power in 0.5-40 Hz band |
| 7 | High-freq noise | ≤ 30% | Power above 40 Hz |
| 8 | Artifacts | ≤ 20% frames | STFT artifact detection |
| 9 | Stationarity | ≥ 0.3 | Time-frequency stability |
| 10 | Lead correlation | ≥ 0.3 | Multi-lead consistency |
| 11 | Spectral flatness | ≤ 0.4 | Wiener entropy (psdpar.m) |
| 12 | Spectral spread | ≤ 25 Hz | f_std maximum (psdpar.m) |
| 13 | Spectral kurtosis | ≥ 1.0 | Spectral peakiness (psdpar.m) |
| 14 | f_mean (centroid) | ≥ 8.5 Hz | Spectral centroid (psdpar.m) |
| 15 | f_std (spread) | ≥ 11.0 Hz | Minimum spread (psdpar.m) |
| 16 | f_iqr (range) | ≥ 7.0 Hz | Frequency IQR (psdpar.m) |

### 7.3 Decision Logic

```python
# Critical rules (must ALL pass)
critical_ok = snr_ok and beat_count_ok and cardiac_power_ok

# Majority rule (70% of all rules must pass)
majority_ok = n_passed >= n_total * 0.7

# Final decision
if critical_ok and majority_ok:
    prediction = 'Acceptable'
else:
    prediction = 'Unacceptable'
```

**Rationale:**
- **Critical rules:** Without SNR, beats, or cardiac content, the ECG is fundamentally unusable
- **Majority rule:** Allows minor issues (e.g., slight baseline wander) if overall quality is good

### 7.4 Classification Result Structure

```python
@dataclass
class ClassificationResult:
    prediction: str              # 'Acceptable' or 'Unacceptable'
    confidence: float            # 0-1 (fraction of rules passed)
    reasons: list[str]           # List of rejection reasons
    feature_flags: dict[str, bool]  # Which rules passed/failed
```

---

## 8. Phase 6: Spectral Parameters (psdpar.m)

### Implementation: `src/spectral_params.py`

Complete Python implementation of the MATLAB `psdpar.m` function from BSA course materials.

**Reference:** Abel Torres, IBEC, October 2024 - BSA Course Chapter 4

### 8.1 All 12 Spectral Parameters

| Parameter | Formula | Description |
|-----------|---------|-------------|
| `f_peak` | `f[argmax(Pxx)]` | Peak frequency (maximum PSD) |
| `f_mean` | `Σ(f × Pxx) / Σ(Pxx)` | Spectral centroid |
| `f_median` | `f at 50% cumulative power` | Median frequency |
| `f_q25` | `f at 25% cumulative power` | First quartile frequency |
| `f_q75` | `f at 75% cumulative power` | Third quartile frequency |
| `f_max95` | `f at 95% cumulative power` | Effective bandwidth |
| `f_std` | `√(Σ((f-f_mean)² × Pxx) / Σ(Pxx))` | Spectral spread |
| `f_iqr` | `f_q75 - f_q25` | Interquartile range |
| `hShannon` | `-Σ(p × log(p)) / log(N)` | Normalized Shannon entropy |
| `spectral_asymmetry` | `μ₃ / σ³` | Spectral skewness |
| `spectral_kurtosis` | `μ₄ / σ⁴ - 3` | Spectral kurtosis (excess) |
| `spectral_flatness` | `geometric_mean(Pxx) / arithmetic_mean(Pxx)` | Wiener entropy |

### 8.2 Theoretical Interpretation

**Good ECG (Acceptable):**
- Concentrated power in cardiac band (0.5-40 Hz)
- Low spectral flatness (periodic QRS pattern)
- Positive kurtosis (sharp spectral peaks)
- Higher f_mean (~10 Hz, cardiac content dominates)

**Bad ECG (Unacceptable):**
- Spread power across frequencies
- High spectral flatness (noise-like, approaching white noise)
- Near-zero kurtosis (flat spectrum)
- Lower f_mean (~6 Hz, drift-dominated)

### 8.3 Implementation

```python
def compute_psd_parameters(f: np.ndarray, Pxx: np.ndarray) -> dict:
    """
    Compute complete PSD parameters from frequency and power arrays.

    Args:
        f: Frequency array (Hz) from periodogram/welch
        Pxx: Power spectral density array

    Returns:
        Dictionary with 12 spectral parameters
    """
    # Normalize PSD to probability distribution
    Pxx_norm = Pxx / np.sum(Pxx)

    # Spectral centroid (mean frequency)
    f_mean = np.sum(f * Pxx_norm)

    # Spectral spread (standard deviation)
    f_std = np.sqrt(np.sum((f - f_mean)**2 * Pxx_norm))

    # Higher moments for shape parameters
    spectral_asymmetry = np.sum((f - f_mean)**3 * Pxx_norm) / f_std**3
    spectral_kurtosis = np.sum((f - f_mean)**4 * Pxx_norm) / f_std**4 - 3

    # ... (other parameters)
```

---

## 9. Phase 7: Results

### 9.1 Dataset Summary

| Property | Value |
|----------|-------|
| Source | PhysioNet/CinC Challenge 2011 |
| Total records | 998 |
| Acceptable | 773 (77.5%) |
| Unacceptable | 225 (22.5%) |
| Duration | 10 seconds |
| Sampling rate | 500 Hz |
| Leads | 12-lead ECG |

### 9.2 Classifier Performance

**Confusion Matrix:**
```
                    Predicted
                  Accept  Reject
  Actual Accept     678      95
  Actual Reject      89     136
```

**Performance Metrics:**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Accuracy | 81.6% | Overall correctness |
| Precision | 88.4% | Of predicted acceptable, % actually acceptable |
| Recall | 87.7% | Of actual acceptable, % correctly identified |
| Specificity | 60.4% | Of actual unacceptable, % correctly rejected |
| F1 Score | 88.1% | Harmonic mean of precision and recall |

### 9.3 Feature Comparison: Good vs Bad ECGs

| Feature | Good ECG (mean ± std) | Bad ECG (mean ± std) |
|---------|----------------------|---------------------|
| SNR (dB) | 10.96 ± 4.12 | 0.68 ± 12.39 |
| Beat count | 12.47 ± 3.10 | 14.75 ± 6.19 |
| Heart rate (bpm) | 74.50 ± 17.43 | 84.94 ± 33.18 |
| RR variability (CV) | 0.08 ± 0.12 | 0.23 ± 0.22 |
| Cardiac power ratio | 0.94 ± 0.08 | 0.63 ± 0.42 |
| f_mean (Hz) | 10.66 ± 3.86 | 6.74 ± 6.77 |
| f_std (Hz) | 14.84 ± 5.88 | 10.29 ± 9.82 |
| f_iqr (Hz) | 9.13 ± 3.46 | 6.00 ± 7.43 |
| Stationarity index | 0.98 ± 0.03 | 0.89 ± 0.29 |

**Key Observations:**
- Good ECGs have significantly higher SNR (10.96 vs 0.68 dB)
- Good ECGs have lower RR variability (0.08 vs 0.23)
- Good ECGs have higher cardiac power ratio (0.94 vs 0.63)
- Good ECGs have higher spectral centroid f_mean (10.66 vs 6.74 Hz)
- Good ECGs are more stationary (0.98 vs 0.89)

### 9.4 Top Rejection Reasons

| Count | Reason |
|-------|--------|
| 81 | Very low SNR (signal dominated by noise) |
| 74 | Low spectral kurtosis (flat spectrum) |
| 73 | Low cardiac frequency content |
| 73 | Low spectral centroid (drift-dominated) |
| 62 | Low inter-lead correlation |
| 17 | Non-stationary signal |
| 16 | Too few beats detected |

---

## 10. Generated Figures

### Available Presentation Figures

| Figure | File | Content |
|--------|------|---------|
| Pan-Tompkins Theory | `pan_tompkins_theory.png` | 6 algorithm steps, good vs bad ECG |
| Spectral Analysis | `spectral_analysis.png` | Time + frequency domain comparison |
| Time-Frequency | `timefreq_analysis.png` | STFT spectrograms |
| Filtering Demo | `filtering_demo.png` | Preprocessing pipeline steps |
| Spectral Parameters | `spectral_parameters_comparison.png` | psdpar.m box plots |
| Spectral Shape | `spectral_shape_scatter.png` | Kurtosis vs asymmetry scatter |
| AR Order Selection | `ar_order_selection.png` | AR model order (arord.m) |

### Figure Descriptions

**1. Pan-Tompkins Theory (`pan_tompkins_theory.png`)**
- 6 rows × 2 columns
- Shows each algorithm step on good and bad ECG
- Demonstrates understanding of QRS detection

**2. Spectral Analysis (`spectral_analysis.png`)**
- Time domain signals + Power Spectral Density
- Shows frequency content differences
- Cardiac frequency peak visible in good ECG

**3. Time-Frequency Analysis (`timefreq_analysis.png`)**
- STFT spectrograms
- Shows temporal evolution of frequency content
- Artifacts visible as sudden power changes

**4. Filtering Demo (`filtering_demo.png`)**
- Before/after comparison
- Shows effect of bandpass filtering
- Uses acceptable ECG for clear demonstration

**5. Spectral Parameters Comparison (`spectral_parameters_comparison.png`)**
- Box plots comparing all psdpar.m parameters
- Green = Acceptable, Red = Unacceptable
- Shows discriminative power of spectral features

**6. Spectral Shape Scatter (`spectral_shape_scatter.png`)**
- Kurtosis vs asymmetry scatter plot
- Shows class separation in 2D feature space
- Demonstrates psdpar.m effectiveness

**7. AR Order Selection (`ar_order_selection.png`)**
- AR model order selection using AIC/BIC
- Based on arord.m from BSA course
- Shows optimal order determination

---

## 11. References

### Primary References
1. Pan & Tompkins (1985). "A Real-Time QRS Detection Algorithm". IEEE Trans. Biomed. Eng.
2. PhysioNet/CinC Challenge 2011. "Improving the Quality of ECGs Collected using Mobile Phones"
3. AHA Guidelines for ECG Bandwidth Standards

### Challenge 2011 Winning Approaches
4. Xia et al. - Spectral features
5. Li & Clifford - Statistical moments
6. Hayn et al. - SNR-based classification
7. Kalkstein et al. - Morphological features
8. Maan et al. - Lead correlation

### BSA Course Materials
9. PSB_master/pan_tompkin.m - Pan-Tompkins MATLAB implementation
10. PSB_master/estimate_cardiac_frequency.m - Fourier/Burg HR estimation
11. Analysis_of_nonstationary_signals/TFRspectrogram.m - STFT analysis
12. tips_PSD_estimation.m - Spectral analysis methods

---

## Summary

### What We Built

| Phase | Module | Purpose |
|-------|--------|---------|
| 1 | Project Setup | Directory structure, dependencies |
| 2 | `data_loader.py` | Load PhysioNet ECG records |
| 3 | `filters.py` | Bandpass, highpass, notch filters |
| 3 | `pan_tompkins.py` | QRS detection algorithm |
| 3 | `cardiac_frequency.py` | Fourier/Burg HR estimation |
| 3 | `qrs_detection.py` | R-peak detection wrapper |
| 3 | `preprocessing.py` | Combined pipeline |
| 4 | `features.py` | 7 feature categories, 25+ features |
| 5 | `classifier.py` | 16-rule quality classifier |
| 5 | `spectral_params.py` | psdpar.m implementation (12 parameters) |

### Final Results

| Metric | Value |
|--------|-------|
| Accuracy | 81.6% |
| Precision | 88.4% |
| Recall | 87.7% |
| Specificity | 60.4% |
| F1 Score | 88.1% |

### Key Theoretical Concepts Demonstrated

1. **Digital Filtering:** Butterworth filters, zero-phase filtering, notch filters
2. **QRS Detection:** Pan-Tompkins algorithm with adaptive thresholding
3. **Spectral Analysis:** Periodogram, AR modeling, psdpar.m parameters
4. **Time-Frequency Analysis:** STFT, stationarity measures
5. **Statistical Analysis:** Higher-order moments, coefficient of variation
6. **Signal Quality:** SNR, baseline wander, artifact detection
7. **Rule-Based Classification:** Interpretable thresholds, decision logic


