# ECG Quality Assessment

**PhysioNet/CinC Challenge 2011** | Binary Classification of 12-Lead ECGs

**Course:** Biomedical Signal Analysis (BSA)

---

## Quick Start (For Team Members)

### 1. Setup Environment

```bash
# Clone/download the project, then:
cd D:\Signals_Project

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_dataset.py
```

This downloads ~103 MB of ECG data to `data/challenge2011/set-a/`.

### 3. Verify Installation

```bash
python -c "from src.data_loader import ECGDataset; d = ECGDataset('data/challenge2011/set-a'); print(f'Loaded {len(d)} records')"
```

Expected output: `Loaded 998 records`

### 4. Generate Figures

```bash
python scripts/generate_figures.py
```

Figures saved to `results/figures/`.

### 5. Run the Classifier

```bash
python -c "
from src.data_loader import ECGDataset
from src.classifier import QualityClassifier

dataset = ECGDataset('data/challenge2011/set-a')
classifier = QualityClassifier()
results = classifier.evaluate(dataset, verbose=True)

print(f'Accuracy: {results[\"accuracy\"]*100:.1f}%')
"
```

---

## Usage Examples

**To run these examples**, start Python interactive mode from the project directory:

```bash
cd D:\Signals_Project
venv\Scripts\activate
python
```

You'll see the `>>>` prompt. Type `exit()` to leave Python when done.

### Classify a Single ECG

```python
import sys
sys.path.insert(0, 'D:/Signals_Project')

from src.data_loader import ECGDataset
from src.classifier import QualityClassifier

# Load dataset
dataset = ECGDataset('data/challenge2011/set-a')

# Create classifier
classifier = QualityClassifier()

# Get an ECG record
ecg = dataset['1002867']  # By record ID
# or: ecg = dataset[0]    # By index

# Classify it
result = classifier.classify(ecg.signal, ecg.fs)

# Check result
print(f"Prediction: {result.prediction}")      # 'Acceptable' or 'Unacceptable'
print(f"Confidence: {result.confidence:.1%}")  # e.g., '81.2%'
print(f"Reasons: {result.reasons}")            # List of rejection reasons
```

### Extract Features from an ECG

```python
from src.features import extract_all_features

ecg = dataset['1002867']
features = extract_all_features(ecg.signal, ecg.fs)

# Access feature groups
print(f"SNR: {features.snr['snr_db']:.1f} dB")
print(f"Heart Rate: {features.morphological['heart_rate']:.0f} bpm")
print(f"Beat Count: {features.morphological['beat_count']}")
print(f"Spectral Flatness: {features.spectral['spectral_flatness']:.3f}")
```

### Evaluate on Full Dataset

```python
from src.data_loader import ECGDataset
from src.classifier import QualityClassifier

dataset = ECGDataset('data/challenge2011/set-a')
classifier = QualityClassifier()

# Evaluate (takes ~5 minutes)
results = classifier.evaluate(dataset, verbose=True)

# Print metrics
print(f"Accuracy:    {results['accuracy']*100:.1f}%")
print(f"Precision:   {results['precision']*100:.1f}%")
print(f"Recall:      {results['recall']*100:.1f}%")
print(f"Specificity: {results['specificity']*100:.1f}%")
print(f"F1 Score:    {results['f1_score']*100:.1f}%")
```

---

## Project Overview

This project classifies 12-lead ECG recordings as **Acceptable** (suitable for clinical interpretation) or **Unacceptable** (too noisy/corrupted for diagnosis).

### Pipeline

```
Raw ECG → Preprocessing → Feature Extraction → Classification → Quality Label
```

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | PhysioNet Challenge 2011 (set-a) |
| **Records** | 998 |
| **Acceptable** | 773 (77.5%) |
| **Unacceptable** | 225 (22.5%) |
| **Duration** | 10 seconds |
| **Sampling Rate** | 500 Hz |
| **Leads** | 12-lead ECG |

---

## Classification Rules (16 Total)

| # | Rule | Threshold | Description |
|---|------|-----------|-------------|
| 1 | SNR | ≥ 5.0 dB | Signal-to-noise ratio |
| 2 | Beat count | 5-30 beats | Expected for 10s recording |
| 3 | Heart rate | 30-200 bpm | Physiological limits |
| 4 | RR regularity | CV ≤ 0.5 | Rhythm consistency |
| 5 | Baseline wander | ratio ≤ 0.3 | Low-frequency drift |
| 6 | Cardiac power | ≥ 50% | Power in 0.5-40 Hz band |
| 7 | High-freq noise | ≤ 30% | Power above 40 Hz |
| 8 | Artifacts | ≤ 20% frames | STFT artifact detection |
| 9 | Stationarity | ≥ 0.3 | Time-frequency stability |
| 10 | Lead correlation | ≥ 0.3 | Multi-lead consistency |
| 11 | Spectral flatness | ≤ 0.4 | Wiener entropy (psdpar.m) |
| 12 | Spectral spread | ≤ 25 Hz | Maximum f_std (psdpar.m) |
| 13 | Spectral kurtosis | ≥ 1.0 | Spectral peakiness (psdpar.m) |
| 14 | f_mean | ≥ 8.5 Hz | Spectral centroid (psdpar.m) |
| 15 | f_std | ≥ 11.0 Hz | Minimum spread (psdpar.m) |
| 16 | f_iqr | ≥ 7.0 Hz | Frequency IQR (psdpar.m) |

**Decision Logic:**
- Critical rules (SNR, beat count, cardiac power) must ALL pass
- At least 70% of all rules must pass
- If both conditions met → **Acceptable**, otherwise → **Unacceptable**

---

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 81.6% |
| **Precision** | 88.4% |
| **Recall** | 87.7% |
| **Specificity** | 60.4% |
| **F1 Score** | 88.1% |

### Confusion Matrix

```
                    Predicted
                  Accept  Reject
  Actual Accept     678      95
  Actual Reject      89     136
```

### Generated Figures

| Figure | File | Description |
|--------|------|-------------|
| Pan-Tompkins | `pan_tompkins_theory.png` | QRS detection steps |
| Spectral Analysis | `spectral_analysis.png` | PSD comparison |
| Time-Frequency | `timefreq_analysis.png` | STFT spectrograms |
| Filtering | `filtering_demo.png` | Filter effects |
| Spectral Parameters | `spectral_parameters_comparison.png` | psdpar.m box plots |
| Spectral Shape | `spectral_shape_scatter.png` | Kurtosis vs asymmetry |
| AR Order | `ar_order_selection.png` | AR model order selection |

---

## Project Structure

```
D:/Signals_Project/
├── src/                          # Python modules
│   ├── data_loader.py            # ECG loading utilities
│   ├── filters.py                # Signal filtering
│   ├── pan_tompkins.py           # QRS detection (Pan-Tompkins)
│   ├── qrs_detection.py          # R-peak detection wrapper
│   ├── cardiac_frequency.py      # HR estimation (Fourier/Burg)
│   ├── features.py               # Feature extraction (7 categories)
│   ├── spectral_params.py        # psdpar.m implementation (12 params)
│   ├── classifier.py             # Rule-based classifier (16 rules)
│   ├── ar_order.py               # AR model order selection
│   └── preprocessing.py          # Combined pipeline
├── scripts/
│   ├── download_dataset.py       # Download PhysioNet data
│   └── generate_figures.py       # Create presentation figures
├── data/
│   └── challenge2011/set-a/      # ECG records (downloaded)
├── docs/
│   └── project_explanation.md    # Full technical documentation
├── results/
│   ├── figures/                  # Generated plots
│   └── classification_results.txt # Numerical results
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"

Add the project root to your Python path:
```python
import sys
sys.path.insert(0, 'D:/Signals_Project')
```

Or run from the project directory:
```bash
cd D:\Signals_Project
python -c "from src.classifier import QualityClassifier; print('OK')"
```

### "FileNotFoundError: Labels not found"

Make sure you downloaded the dataset:
```bash
python scripts/download_dataset.py
```

And use the correct path:
```python
dataset = ECGDataset('data/challenge2011/set-a')  # Relative path
# or
dataset = ECGDataset('D:/Signals_Project/data/challenge2011/set-a')  # Absolute path
```

### "Virtual environment not activated"

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

## Key Modules

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `data_loader.py` | Load ECG data | `ECGDataset`, `ECGRecord` |
| `classifier.py` | Quality classification | `QualityClassifier.classify()` |
| `features.py` | Feature extraction | `extract_all_features()` |
| `spectral_params.py` | PSD parameters | `compute_psd_parameters()` |
| `pan_tompkins.py` | QRS detection | `pan_tompkins()` |
| `filters.py` | Signal filtering | `bandpass_filter()`, `notch_filter()` |

---

## Requirements

- Python 3.8+
- NumPy
- SciPy
- WFDB (PhysioNet data access)
- Matplotlib
- Pandas

Install all:
```bash
pip install -r requirements.txt
```

---

## References

1. **Pan & Tompkins (1985)** - "A Real-Time QRS Detection Algorithm", IEEE Trans. Biomed. Eng.
2. **PhysioNet Challenge 2011** - https://physionet.org/content/challenge-2011/
3. **psdpar.m** - Abel Torres, IBEC, 2024 (BSA Course)

---

## Documentation

For detailed technical explanation, see:
- `docs/project_explanation.md` - Complete implementation details
- `results/classification_results.txt` - All numerical results

---

*BSA Course Project - December 2025*
