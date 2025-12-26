# Biomedical Signal Analysis: Theory for ECG Quality Classification

**Author:** Rocío Ávalos
**Course:** Biomedical Signal Analysis (BSA) - UPC BarcelonaTech
**Project:** PhysioNet/CinC Challenge 2011 - ECG Quality Classification

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Preprocessing: Filtering for Artifact Removal](#2-preprocessing-filtering-for-artifact-removal)
3. [QRS Detection: Pan-Tompkins Algorithm](#3-qrs-detection-pan-tompkins-algorithm)
4. [Frequency-Domain Characterization](#4-frequency-domain-characterization)
5. [Heart Rate Variability (HRV) Analysis](#5-heart-rate-variability-hrv-analysis)
6. [Signal Quality Metrics](#6-signal-quality-metrics)
7. [Time-Frequency Analysis](#7-time-frequency-analysis)
8. [Classification Methods for ECG Quality](#8-classification-methods-for-ecg-quality)
9. [References](#9-references)

---

## Implementation Cross-Reference

| Theory Section | Python Module | MATLAB Source |
|----------------|---------------|---------------|
| Filtering | `src/filters.py` | `baseline_ECG.m`, `notch.m` |
| QRS Detection | `src/pan_tompkins.py` | `pan_tompkin.m` |
| Cardiac Frequency | `src/cardiac_frequency.py` | `estimate_cardiac_frequency.m` |
| Feature Extraction | `src/features.py` | `psdpar.m`, `powerband.m` |
| Classification | `src/classifier.py`, `src/classifier_optimized.py` | - |

---

## 1. Introduction

### 1.1 ECG Signal Characteristics

The electrocardiogram (ECG) is a recording of the electrical activity of the heart. A standard **12-lead ECG** provides a comprehensive view of cardiac electrical activity from different angles:

| Lead Type | Leads | View |
|-----------|-------|------|
| Limb leads | I, II, III | Frontal plane |
| Augmented leads | aVR, aVL, aVF | Frontal plane |
| Precordial leads | V1-V6 | Horizontal plane |

**PhysioNet Challenge 2011 Data Specifications:**
- **Sampling rate:** 500 Hz
- **Resolution:** 16 bits, 5 µV/bit
- **Duration:** 10 seconds per recording
- **Bandwidth:** 0.05-100 Hz (diagnostic)
- **Dataset:** 998 records (773 Acceptable, 225 Unacceptable)

### 1.2 The Quality Classification Problem

ECG quality assessment determines whether a recording is **acceptable** (interpretable for diagnosis) or **unacceptable** (too corrupted by artifacts/noise).

**Quality Grades (Challenge 2011):**

| Grade | Score | Description |
|-------|-------|-------------|
| A | 0.95 | Outstanding - no visible noise/artifact |
| B | 0.85 | Good - transient artifact, low-level noise |
| C | 0.75 | Adequate - interpretable despite flaws |
| D | 0.60 | Poor - difficult interpretation, missing signals |
| F | 0.00 | Unacceptable - cannot be interpreted |

**Classification Rule:**
- **Acceptable:** Average grade ≥ 0.7, no more than one F grade
- **Unacceptable:** Average grade < 0.7

### 1.3 Common ECG Artifacts

| Artifact | Source | Frequency Range |
|----------|--------|-----------------|
| Baseline wander | Respiration, movement | < 0.5 Hz |
| Powerline interference | Electrical mains | 50/60 Hz |
| Muscle noise (EMG) | Skeletal muscle | > 30 Hz |
| Motion artifacts | Electrode movement | Broadband |
| Electrode contact | Poor skin contact | Various |

---

## 2. Preprocessing: Filtering for Artifact Removal

**Python Implementation:** `src/filters.py`
**MATLAB Source:** `F:/Master/Signals/Filtering to remove artefacts/`

### 2.1 Baseline Wander Removal

Baseline wander is caused by respiration and patient movement, typically below 0.5 Hz.

#### Method 1: Cascaded Median Filter

A two-stage median filter approach effectively removes baseline wander while preserving the ECG morphology:

```
Stage 1: M1 = 200 ms window → removes high-frequency variations
Stage 2: M2 = 600 ms window → extracts smooth baseline
Corrected ECG = Original ECG - Baseline estimate
```

**From `baseline_ECG.m`:**
```matlab
M1 = floor(200e-3 * Fs);  % 200 ms window
M2 = floor(600e-3 * Fs);  % 600 ms window
% First median pass
for i = 1:L-M1
    m1(i+M1/2) = median(ECG(i:i+M1));
end
% Second median pass
for i = 1:L-M2
    baseline(i+M2/2) = median(m1(i:i+M2));
end
```

**Python equivalent in `src/filters.py`:** `remove_baseline_median()`

#### Method 2: High-Pass Filtering

- **Cutoff frequency:** 0.5 Hz (preserves P-wave)
- **Filter type:** Butterworth (maximally flat passband)
- **Order:** 4 (balance between sharpness and phase distortion)

### 2.2 High-Frequency Noise Removal

**Low-pass filtering** removes muscle noise and high-frequency interference:
- **Diagnostic ECG:** cutoff at 100 Hz
- **Monitoring ECG:** cutoff at 40 Hz
- **Warning:** Aggressive filtering (< 40 Hz) can distort QRS complex

### 2.3 Powerline Interference Removal

**Notch filter** at 50 Hz (Europe) or 60 Hz (Americas):

```python
# From src/filters.py
def notch_filter(signal, fs, freq=50.0, quality_factor=30.0):
    nyq = fs / 2
    w0 = freq / nyq
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, signal)
```

### 2.4 Critical Concept: filter() vs filtfilt()

| Function | Phase | Delay | Use Case |
|----------|-------|-------|----------|
| `filter()` | Non-zero | Yes | Real-time processing |
| `filtfilt()` | Zero-phase | No | Offline diagnostic analysis |

**Important:** For diagnostic ECG analysis, always use `filtfilt()` to preserve timing of P-QRS-T waves.

### 2.5 AHA Recommended ECG Bandwidth

| Application | Low cutoff | High cutoff |
|-------------|------------|-------------|
| Diagnostic | 0.05 Hz | 150 Hz |
| Monitoring | 0.5 Hz | 40 Hz |
| **This project** | **0.5 Hz** | **100 Hz** |

---

## 3. QRS Detection: Pan-Tompkins Algorithm

**Python Implementation:** `src/pan_tompkins.py` (547 lines, enhanced version)
**MATLAB Source:** `pan_tompkin.m` (Hooman Sedghamiz)
**Reference:** Pan & Tompkins, IEEE Trans. Biomed. Eng., 1985

### 3.1 Algorithm Overview

The Pan-Tompkins algorithm is the gold standard for real-time QRS detection, consisting of preprocessing stages followed by adaptive decision rules.

### 3.2 Preprocessing Stages

```
Raw ECG → Bandpass (5-15 Hz) → Derivative → Squaring → Moving Average → Thresholding
```

#### Stage 1: Bandpass Filter (5-15 Hz)
- Removes baseline wander (< 5 Hz)
- Removes high-frequency noise (> 15 Hz)
- Maximizes QRS energy while attenuating P/T waves

#### Stage 2: Derivative Filter
Highlights the steep slopes of the QRS complex:

```
H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^2)
Coefficients: [-1, -2, 0, 2, 1] × (1/8)
```

#### Stage 3: Squaring
Makes all values positive and emphasizes large amplitudes:
```
y[n] = x[n]²
```

#### Stage 4: Moving Window Integration
Smooths the signal to obtain a pulse-like waveform:
```
y[n] = (1/N) × Σ x[n-k] for k = 0 to N-1
```
**Window length:** N = 150 ms (approximately one QRS width)

### 3.3 Decision Rules

#### Adaptive Thresholding

Two thresholds adapt continuously:
```
THR_SIG = NOISE_LEV + 0.25 × (SIG_LEV - NOISE_LEV)
THR_NOISE = 0.5 × THR_SIG
```

Level updates after each peak:
- If peak > THR_SIG: `SIG_LEV = 0.125 × peak + 0.875 × SIG_LEV`
- If THR_NOISE < peak < THR_SIG: `NOISE_LEV = 0.125 × peak + 0.875 × NOISE_LEV`

#### Refractory Period
No QRS can occur within **200 ms** of a previous detection (physiological constraint).

#### T-Wave Discrimination
If peak occurs within **360 ms** of previous QRS:
- Compare slope to previous QRS slope
- If slope < 0.5 × previous slope → T-wave (reject)
- Otherwise → valid QRS

#### Searchback
If no QRS detected for **1.66 × mean RR interval**:
- Search back for peaks between THR_NOISE and THR_SIG
- Accept highest peak as missed QRS

### 3.4 Enhanced Implementation Features

**Your `src/pan_tompkins.py` includes:**
- Flat signal detection (sleepecg approach)
- Ricker wavelet filter option (XQRS approach)
- Learning phase for threshold calibration
- Backsearch mechanism for missed beats

---

## 4. Frequency-Domain Characterization

**Python Implementation:** `src/cardiac_frequency.py`, `src/features.py`
**MATLAB Source:** `F:/Master/Signals/Frequency-domain Characterization/C_4/`

### 4.1 PSD Estimation Methods

#### Welch Periodogram (Non-parametric)

Divides signal into overlapping segments, applies windowing, and averages periodograms.

**Parameters:**
- **Segment length:** 2-4 seconds for ECG
- **Overlap:** 50% (standard)
- **Window:** Hann (good frequency resolution)
- **Variance reduction:** Factor of 1/K where K = number of segments

```python
from scipy.signal import welch
f, Pxx = welch(signal, fs, nperseg=segment_length, noverlap=overlap)
```

**Best for:** Signals longer than a few seconds

#### AR Model-Based (Burg's Method)

Parametric approach assuming signal is output of an autoregressive model.

**Advantages:**
- Sharper spectral peaks
- Better for short signals (< 1 second)
- Always stable (poles inside unit circle)

**Model Order Selection** (from `arord.m`):

| Criterion | Formula | Characteristic |
|-----------|---------|----------------|
| FPE | ρ × (1+p/N)/(1-p/N) | Balances fit and complexity |
| AIC | N×ln(ρ) + 2p | Penalizes complexity |
| AICm | N×ln(ρ) + p×ln(N) | Stronger penalty (recommended) |

**Your implementation:** `src/cardiac_frequency.py` includes `_arburg()` with Levinson recursion.

### 4.2 PSD Parameters for Quality Assessment

**MATLAB Reference:** `psdpar.m` (Abel Torres, IBEC, 2024)

#### Frequency Descriptors

| Parameter | Definition | Formula |
|-----------|------------|---------|
| **f_peak** | Peak frequency | argmax(Pxx) |
| **f_mean** | Mean frequency | Σ(f×Pxx) / Σ(Pxx) |
| **f_median** | Median frequency | f where cumsum(Pxx) = 0.5 |
| **f_q25** | First quartile | f where cumsum(Pxx) = 0.25 |
| **f_q75** | Third quartile | f where cumsum(Pxx) = 0.75 |
| **f_max95** | 95% frequency | f where cumsum(Pxx) = 0.95 |

#### Dispersion Parameters

| Parameter | Definition | Formula |
|-----------|------------|---------|
| **f_std** | Spectral spread | √(Σ((f-f_mean)²×Pxx) / Σ(Pxx)) |
| **f_iqr** | Interquartile range | f_q75 - f_q25 |

#### Shape Parameters (Key for Quality Classification)

| Parameter | Formula | Interpretation |
|-----------|---------|----------------|
| **h_Shannon** | -Σ(p×ln(p)) / ln(N) | **1 = flat (noise)**, **0 = narrow (clean)** |
| **c_Asymmetry** | μ₃ / σ³ | Spectral skewness |
| **c_Kurtosis** | μ₄ / σ⁴ - 3 | Tail heaviness |

Where p = Pxx/Σ(Pxx) is the normalized PSD.

**Shannon Entropy Interpretation:**
- High entropy (~1) → flat, noise-like spectrum → **poor quality**
- Low entropy (~0) → concentrated spectrum → **good quality**

**Your implementation:** `src/features.py` → `extract_spectral_features()` computes `spectral_entropy`

---

## 5. Heart Rate Variability (HRV) Analysis

**MATLAB Source:** `F:/Master/Signals/Frequency-domain Characterization/C_4/`
**Reference:** `powerband.m` for band power calculation

### 5.1 RR Interval Extraction

1. Detect R-peaks using Pan-Tompkins
2. Calculate RR intervals: RR[i] = R[i+1] - R[i]
3. Remove ectopic beats (interpolate if |RR - mean_RR| > 20%)
4. Resample to uniform time grid (typically 4 Hz)

**Your implementation:** `src/qrs_detection.py` → `compute_rr_intervals()`

### 5.2 HRV Frequency Bands

| Band | Frequency Range | Physiological Meaning |
|------|-----------------|----------------------|
| **VLF** | 0.003 - 0.04 Hz | Thermoregulation, hormonal regulation |
| **LF** | 0.04 - 0.15 Hz | Sympathetic + Parasympathetic activity |
| **HF** | 0.15 - 0.40 Hz | Parasympathetic (vagal) - respiratory sinus arrhythmia |

### 5.3 HRV Metrics

**Absolute Power:**
```
P_band = ∫ PSD(f) df  [ms²]
```

**Normalized Units:**
```
LF_nu = LF / (LF + HF) × 100
HF_nu = HF / (LF + HF) × 100
```

**LF/HF Ratio:**
- Indicator of sympathovagal balance
- Normal: 1.5 - 2.0
- High: Sympathetic dominance
- Low: Parasympathetic dominance

**Total Power:**
```
TP = VLF + LF + HF  [ms²]
```

### 5.4 HRV as ECG Quality Indicator

| Indicator | Good Quality | Poor Quality |
|-----------|--------------|--------------|
| RR intervals | Regular, physiological | Erratic, impossible values |
| Total power | Within normal range | Abnormally high/low |
| Missing beats | Rare | Frequent |
| RR range | 0.3-2.0 s | Values outside range |

**Physiological constraints:**
- Minimum RR: ~300 ms (HR max ~200 bpm)
- Maximum RR: ~2000 ms (HR min ~30 bpm)
- Maximum RR change: < 20% beat-to-beat

**Your implementation:** `src/features.py` → `extract_morphological_features()` uses RR variability (CV)

---

## 6. Signal Quality Metrics

**Python Implementation:** `src/features.py` (592 lines, 7 feature categories)

### 6.1 Signal-to-Noise Ratio (SNR)

**Definition:**
```
SNR = 10 × log₁₀(P_signal / P_noise)  [dB]
```

**Your implementation:** `extract_snr_features()` extracts from QRS windows vs. noise windows

| Quality | SNR Range |
|---------|-----------|
| Good | > 10 dB |
| Acceptable | 5-10 dB |
| Poor | < 5 dB |

### 6.2 Baseline Wander Features

**From `extract_baseline_features()`:**
- `baseline_power` - Power in baseline signal
- `baseline_std` - Standard deviation of baseline
- `baseline_range` - Peak-to-peak baseline variation
- `baseline_power_ratio` - Ratio to total signal power

High baseline power ratio suggests poor electrode contact.

### 6.3 Statistical Features

**From `extract_statistical_features()`:**
| Feature | Interpretation |
|---------|----------------|
| Skewness | Asymmetry of amplitude distribution |
| Kurtosis | Presence of outliers/artifacts |
| Zero-crossing rate | Signal complexity |

High kurtosis often indicates artifact spikes.

### 6.4 Feature Categories in Your Implementation

| Category | Features | Function |
|----------|----------|----------|
| SNR | 3 features | `extract_snr_features()` |
| Baseline | 4 features | `extract_baseline_features()` |
| Morphological | 5 features | `extract_morphological_features()` |
| Spectral | 5 features | `extract_spectral_features()` |
| Statistical | 8 features | `extract_statistical_features()` |
| Time-frequency | 4 features | `extract_timefreq_features()` |
| Lead correlation | 3 features | `extract_lead_correlation()` |

**Total: 30+ features** extracted in `extract_all_features()`

---

## 7. Time-Frequency Analysis

**Python Implementation:** `src/features.py` → `extract_timefreq_features()`
**MATLAB Source:** `F:/Master/Signals/Analysis of Nonstationary Signals/`

### 7.1 Short-Time Fourier Transform (STFT)

**Spectrogram:**
```
S(t,f) = |STFT{x(τ)}|² = |∫ x(τ) w(τ-t) e^(-j2πfτ) dτ|²
```

**Your parameters:**
- Window: 500 ms
- Overlap: 75%
- FFT points: Power of 2

### 7.2 Time-Frequency Features

| Feature | Description | Quality Indicator |
|---------|-------------|-------------------|
| `temporal_variance` | Variance across time frames | High = non-stationary artifacts |
| `spectral_flatness` | Flatness of spectrum | High = noise-like |
| `artifact_frame_ratio` | Fraction of artifact frames | > 0.2 = poor quality |
| `stationarity_index` | Signal stationarity | < 0.3 = non-stationary |

### 7.3 Alternative TFR Methods

| Method | Characteristics |
|--------|-----------------|
| **Spectrogram (STFT)** | Linear, no cross-terms, limited resolution |
| **Scalogram (Wavelet)** | Multi-resolution, good for transients |
| **Wigner-Ville** | Highest resolution, but cross-terms |

For ECG quality: **Spectrogram is sufficient** (simple, interpretable)

---

## 8. Classification Methods for ECG Quality

**Python Implementation:** `src/classifier.py`, `src/classifier_optimized.py`

### 8.1 Feature Engineering

**Feature Categories in Your Implementation:**

| Category | Features | Source |
|----------|----------|--------|
| Time-domain | Mean, std, min, max | Raw ECG |
| Spectral | Entropy, kurtosis, skewness | `psdpar.m` concepts |
| QRS-based | Detection rate, RR variability | `pan_tompkins.py` |
| Morphological | QRS amplitude consistency | Signal analysis |
| Lead correlation | Inter-lead correlations | Multi-lead analysis |

### 8.2 Your Rule-Based Classifier

**From `src/classifier.py` - 10 Quality Rules:**

| Rule | Threshold | Rationale |
|------|-----------|-----------|
| SNR | ≥ 5 dB | Minimum signal quality |
| Beat count | 5-30 (10s) | Physiologically valid |
| Heart rate | 30-200 bpm | Normal range |
| RR regularity | CV ≤ 0.5 | Rhythm stability |
| Baseline wander | Ratio ≤ 0.3 | Electrode contact |
| Cardiac power | ≥ 50% in 0.5-40 Hz | Valid cardiac content |
| High-freq noise | ≤ 30% above 40 Hz | Noise level |
| Artifacts | ≤ 20% of frames | Artifact burden |
| Stationarity | Index ≥ 0.3 | Signal consistency |
| Lead correlation | Mean ≥ 0.3 | Multi-lead consistency |

**Classification Logic:**
- Critical rules (SNR, beat_count, cardiac_power) must all pass
- Majority rule: ≥70% of all rules must pass
- Confidence: n_passed / n_total

### 8.3 Alternative Classification Algorithms

#### Decision Trees
- Interpretable rules (important for clinical applications)
- Feature importance ranking
- Prone to overfitting → use pruning or ensemble methods

#### Support Vector Machines (SVM)
- Effective for high-dimensional feature spaces
- Kernel selection: RBF (default), linear, polynomial
- Hyperparameter tuning: C (regularization), γ (RBF kernel)
- **Best performer in PhysioNet 2011** (Li & Clifford, 92.6%)

#### Random Forests
- Ensemble of decision trees
- Built-in feature importance
- Robust to overfitting

#### Heuristic Rules (Your Approach)
- Rule-based thresholds
- Fastest execution time
- Good for real-time mobile applications
- **Interpretable** - can explain why an ECG was rejected

### 8.4 Your Optimized Classifier

**From `src/classifier_optimized.py`:**

1. **Grid Search Optimization:** Tests 288 threshold combinations
2. **Enhanced Features:** Adds cardiac frequency confidence
3. **Multi-Lead Consensus:** Acceptable if ≥7/12 leads pass
4. **Train/Val/Test Split:** 60/20/20 with stratification

### 8.5 Evaluation Protocol

**Metrics:**

| Metric | Formula | Your Implementation |
|--------|---------|---------------------|
| Accuracy | (TP + TN) / Total | `classifier.evaluate()` |
| Sensitivity | TP / (TP + FN) | Detecting bad ECGs |
| Specificity | TN / (TN + FP) | Not rejecting good ECGs |
| F1-Score | 2×(Prec×Rec)/(Prec+Rec) | Balanced metric |

**Validation Strategy:**
- K-fold cross-validation (k=5 or k=10)
- Stratified splits (maintain class balance)

### 8.6 Challenge-Winning Approaches (Benchmark)

| Team | Accuracy | Key Features | Classifier |
|------|----------|--------------|------------|
| **Xia et al.** | 93.2% | Entropy, multi-stage | Decision rules |
| **Li & Clifford** | 92.6% | Higher-order moments, intra-lead | SVM |
| **Hayn et al.** | 87.3% | SNR, real-time | Fast heuristics |
| **Moody** | 89.6% | Rule-based | Heuristic rules |

**Key insights:**
- Entropy was the most discriminative feature
- Combining time and frequency domain features improves performance
- Simple rule-based methods can achieve good results with fast execution

---

## 9. References

### Course Materials
1. Torres, A. (2024). *Biomedical Signal Analysis Course Materials*. Institute for Bioengineering of Catalonia (IBEC), Universitat Politècnica de Catalunya.

### MATLAB Functions (F:/Master/Signals/)
- `psdpar.m` - PSD parameter extraction (Abel Torres, IBEC, 2024)
- `arord.m`, `arspec.m` - AR model estimation (Abel Torres, IBEC, 2024)
- `baseline_ECG.m` - Baseline wander removal
- `pan_tompkin.m` - QRS detection (Hooman Sedghamiz, Linköping University)
- `powerband.m` - Frequency band power calculation
- `TFRspectrogram.m` - Time-frequency analysis

### Python Implementation (D:/Signals_Project/src/)
- `filters.py` - Bandpass, notch, median baseline (177 lines)
- `pan_tompkins.py` - Enhanced QRS detection (547 lines)
- `cardiac_frequency.py` - Fourier + Burg methods (340 lines)
- `features.py` - 7 categories, 30+ features (592 lines)
- `classifier.py` - Rule-based, 10 rules (440 lines)
- `classifier_optimized.py` - Grid search, multi-lead (520 lines)

### Scientific Papers
1. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230-236.

2. Silva, I., Moody, G. B., & Celi, L. (2011). Improving the quality of ECGs collected using mobile phones: The PhysioNet/Computing in Cardiology Challenge 2011. *Computing in Cardiology*, 38, 273-276.

3. Xia, H., et al. (2011). A multistage computer test algorithm for improving the quality of ECGs. *Computing in Cardiology*, 38.

4. Li, Q., & Clifford, G. D. (2011). Signal quality indices and data fusion for determining acceptability of electrocardiograms. *Computing in Cardiology*, 38.

5. Hayn, D., Jammerbund, B., & Schreier, G. (2011). Real-time visualization of signal quality during mobile ECG recording. *Computing in Cardiology*, 38.

6. Moody, B. E. (2011). A rule-based method for ECG quality control. *Computing in Cardiology*, 38.

---

*Document compiled for the PhysioNet/CinC Challenge 2011 project*
*Biomedical Signal Analysis - Master's Program, UPC BarcelonaTech*
*December 2025*
