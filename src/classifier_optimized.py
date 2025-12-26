"""
Optimized ECG Quality Classifier

Improvements over basic classifier:
1. Grid search threshold optimization
2. Proper train/val/test split
3. Cardiac frequency feature
4. Spectral quality (peak-to-mean ratio)
5. Multi-lead consensus

Target: 85%+ accuracy (vs 72.5% baseline)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .features import extract_all_features, ECGFeatures
from .cardiac_frequency import estimate_cardiac_frequency_fourier


# =============================================================================
# DATA SPLITTING
# =============================================================================

def create_train_val_test_split(dataset, test_size=0.2, val_size=0.2, random_state=42):
    """
    Create proper train/validation/test split.

    Split: 60% train, 20% validation, 20% test
    Stratified by quality label.

    Args:
        dataset: ECGDataset instance
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed

    Returns:
        (train_ids, val_ids, test_ids)
    """
    all_ids = dataset.record_ids
    labels = [dataset._quality_map.get(rid, 'Unknown') for rid in all_ids]

    # First split: train+val vs test
    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        all_ids, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=random_state
    )

    return train_ids, val_ids, test_ids


# =============================================================================
# ENHANCED FEATURE EXTRACTION
# =============================================================================

def extract_enhanced_features(signal: np.ndarray, fs: float) -> dict:
    """
    Extract enhanced features including cardiac frequency and spectral quality.

    Args:
        signal: ECG signal (1D for single lead)
        fs: Sampling frequency

    Returns:
        Dictionary with all features
    """
    from scipy.signal import welch

    # Basic features
    base_features = extract_all_features(signal.reshape(-1, 1) if signal.ndim == 1 else signal, fs)
    features = base_features.to_dict()

    # Single lead for additional features
    if signal.ndim == 2:
        single_lead = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
    else:
        single_lead = signal.flatten()

    # ===========================================
    # NEW: Cardiac Frequency Feature
    # ===========================================
    try:
        fc_result = estimate_cardiac_frequency_fourier(single_lead, fs)
        features['cardiac_freq_hz'] = fc_result.fcard if not np.isnan(fc_result.fcard) else 0.0
        features['cardiac_freq_bpm'] = fc_result.hr if not np.isnan(fc_result.hr) else 0.0
        features['cardiac_freq_confidence'] = fc_result.confidence
    except:
        features['cardiac_freq_hz'] = 0.0
        features['cardiac_freq_bpm'] = 0.0
        features['cardiac_freq_confidence'] = 0.0

    # ===========================================
    # NEW: Spectral Quality Features
    # ===========================================
    try:
        f, Pxx = welch(single_lead, fs=fs, nperseg=int(2*fs), noverlap=int(fs))

        # Cardiac band (0.5-2.5 Hz = 30-150 bpm)
        cardiac_mask = (f >= 0.5) & (f <= 2.5)

        if np.any(cardiac_mask):
            cardiac_peak = np.max(Pxx[cardiac_mask])
            cardiac_mean = np.mean(Pxx[cardiac_mask])
            features['peak_to_mean_ratio'] = cardiac_peak / (cardiac_mean + 1e-10)
        else:
            features['peak_to_mean_ratio'] = 1.0

    except:
        features['peak_to_mean_ratio'] = 1.0

    return features


# =============================================================================
# OPTIMIZED THRESHOLDS
# =============================================================================

@dataclass
class OptimizedThresholds:
    """Thresholds found via grid search optimization."""
    # Core thresholds (to be optimized)
    snr_min: float = 5.0
    beat_count_min: int = 5
    beat_count_max: int = 25
    rr_cv_max: float = 0.5
    cardiac_power_ratio_min: float = 0.5

    # Additional thresholds
    cardiac_freq_min: float = 0.5    # 30 bpm
    cardiac_freq_max: float = 3.0    # 180 bpm
    peak_to_mean_min: float = 2.0    # Need clear spectral peak


def classify_with_thresholds(features: dict, thresholds: OptimizedThresholds) -> tuple:
    """
    Classify using specific thresholds.

    Returns:
        (prediction, reasons) tuple
    """
    reasons = []
    th = thresholds

    # Rule 1: SNR
    snr_db = features.get('snr_snr_db', 0)
    if np.isinf(snr_db) or snr_db < th.snr_min:
        reasons.append(f"Low SNR ({snr_db:.1f} dB)")

    # Rule 2: Beat count
    beat_count = features.get('morph_beat_count', 0)
    if beat_count < th.beat_count_min:
        reasons.append(f"Too few beats ({beat_count})")
    elif beat_count > th.beat_count_max:
        reasons.append(f"Too many beats ({beat_count})")

    # Rule 3: RR variability
    rr_cv = features.get('morph_rr_cv', 0)
    if rr_cv > th.rr_cv_max and beat_count >= 3:
        reasons.append(f"Irregular rhythm (CV={rr_cv:.2f})")

    # Rule 4: Cardiac power ratio
    cardiac_ratio = features.get('spectral_cardiac_power_ratio', 0)
    if cardiac_ratio < th.cardiac_power_ratio_min:
        reasons.append(f"Low cardiac content ({cardiac_ratio:.0%})")

    # Rule 5: Cardiac frequency (NEW)
    fc_hz = features.get('cardiac_freq_hz', 0)
    if fc_hz > 0 and (fc_hz < th.cardiac_freq_min or fc_hz > th.cardiac_freq_max):
        reasons.append(f"Abnormal cardiac freq ({fc_hz*60:.0f} bpm)")

    # Rule 6: Spectral peak quality (NEW)
    peak_ratio = features.get('peak_to_mean_ratio', 0)
    if peak_ratio < th.peak_to_mean_min:
        reasons.append(f"No clear cardiac peak")

    # Decision: Acceptable if no critical issues
    if len(reasons) == 0:
        prediction = 'Acceptable'
    elif len(reasons) <= 1 and 'Low SNR' not in str(reasons):
        # Allow one minor issue (but not SNR)
        prediction = 'Acceptable'
    else:
        prediction = 'Unacceptable'

    return prediction, reasons


# =============================================================================
# GRID SEARCH OPTIMIZATION
# =============================================================================

def grid_search_thresholds(
    features_list: list[dict],
    labels: list[str],
    verbose: bool = True
) -> tuple:
    """
    Find optimal thresholds using grid search.

    Args:
        features_list: List of feature dictionaries
        labels: List of quality labels ('Acceptable' or 'Unacceptable')
        verbose: Print progress

    Returns:
        (best_thresholds, results_df)
    """
    # Define search space
    snr_thresholds = [3, 4, 5, 6, 7, 8]
    rr_cv_thresholds = [0.3, 0.4, 0.5, 0.6]
    beat_ranges = [(4, 25), (5, 22), (6, 20), (7, 18)]
    cardiac_power_thresholds = [0.3, 0.4, 0.5, 0.6]

    best_accuracy = 0
    best_f1 = 0
    best_thresholds = OptimizedThresholds()
    results = []

    labels_binary = np.array([1 if l == 'Acceptable' else 0 for l in labels])

    total_combinations = len(snr_thresholds) * len(rr_cv_thresholds) * len(beat_ranges) * len(cardiac_power_thresholds)

    if verbose:
        print(f"Testing {total_combinations} threshold combinations...")

    count = 0
    for snr_thr in snr_thresholds:
        for rr_thr in rr_cv_thresholds:
            for beat_min, beat_max in beat_ranges:
                for cardiac_thr in cardiac_power_thresholds:
                    count += 1

                    # Create thresholds
                    th = OptimizedThresholds(
                        snr_min=snr_thr,
                        beat_count_min=beat_min,
                        beat_count_max=beat_max,
                        rr_cv_max=rr_thr,
                        cardiac_power_ratio_min=cardiac_thr
                    )

                    # Test this combination
                    predictions = []
                    for feat in features_list:
                        pred, _ = classify_with_thresholds(feat, th)
                        predictions.append(1 if pred == 'Acceptable' else 0)

                    predictions = np.array(predictions)

                    # Calculate metrics
                    accuracy = accuracy_score(labels_binary, predictions)

                    # Handle edge cases for precision/recall
                    if np.sum(predictions) > 0 and np.sum(labels_binary) > 0:
                        precision = precision_score(labels_binary, predictions, zero_division=0)
                        recall = recall_score(labels_binary, predictions, zero_division=0)
                        f1 = f1_score(labels_binary, predictions, zero_division=0)
                    else:
                        precision = recall = f1 = 0

                    # Specificity
                    tn = np.sum((predictions == 0) & (labels_binary == 0))
                    fp = np.sum((predictions == 1) & (labels_binary == 0))
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

                    results.append({
                        'snr_threshold': snr_thr,
                        'rr_cv_threshold': rr_thr,
                        'beat_min': beat_min,
                        'beat_max': beat_max,
                        'cardiac_power_threshold': cardiac_thr,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'specificity': specificity,
                        'f1': f1
                    })

                    # Update best (prioritize F1 score for balanced performance)
                    if f1 > best_f1 or (f1 == best_f1 and accuracy > best_accuracy):
                        best_f1 = f1
                        best_accuracy = accuracy
                        best_thresholds = th

                    if verbose and count % 50 == 0:
                        print(f"  {count}/{total_combinations} tested... best F1={best_f1:.1%}")

    results_df = pd.DataFrame(results)

    if verbose:
        print(f"\nBest thresholds found:")
        print(f"  SNR min: {best_thresholds.snr_min} dB")
        print(f"  RR CV max: {best_thresholds.rr_cv_max}")
        print(f"  Beat range: {best_thresholds.beat_count_min}-{best_thresholds.beat_count_max}")
        print(f"  Cardiac power min: {best_thresholds.cardiac_power_ratio_min}")
        print(f"  Best accuracy: {best_accuracy:.1%}")
        print(f"  Best F1: {best_f1:.1%}")

    return best_thresholds, results_df


# =============================================================================
# MULTI-LEAD CONSENSUS
# =============================================================================

def multi_lead_quality(ecg_12lead: np.ndarray, fs: float, thresholds: OptimizedThresholds) -> dict:
    """
    Assess quality across all 12 leads.
    ECG is acceptable if majority of leads are good.

    Args:
        ecg_12lead: 12-lead ECG (samples, 12)
        fs: Sampling frequency
        thresholds: Classification thresholds

    Returns:
        Dictionary with prediction and per-lead results
    """
    n_leads = ecg_12lead.shape[1]
    lead_results = []

    for lead_idx in range(n_leads):
        lead_signal = ecg_12lead[:, lead_idx]

        try:
            features = extract_enhanced_features(lead_signal, fs)
            pred, reasons = classify_with_thresholds(features, thresholds)

            lead_results.append({
                'lead_idx': lead_idx,
                'prediction': pred,
                'reasons': reasons,
                'acceptable': pred == 'Acceptable'
            })
        except:
            lead_results.append({
                'lead_idx': lead_idx,
                'prediction': 'Error',
                'reasons': ['Processing error'],
                'acceptable': False
            })

    # Count acceptable leads
    acceptable_count = sum(r['acceptable'] for r in lead_results)

    # Decision: Need at least 7/12 leads acceptable (majority)
    min_acceptable = 7

    if acceptable_count >= min_acceptable:
        prediction = 'Acceptable'
        confidence = acceptable_count / n_leads
    else:
        prediction = 'Unacceptable'
        confidence = 1 - (acceptable_count / n_leads)

    return {
        'prediction': prediction,
        'confidence': confidence,
        'acceptable_leads': acceptable_count,
        'total_leads': n_leads,
        'per_lead_results': lead_results
    }


# =============================================================================
# OPTIMIZED CLASSIFIER
# =============================================================================

class OptimizedClassifier:
    """
    Optimized ECG quality classifier with:
    - Grid-search tuned thresholds
    - Enhanced features (cardiac frequency, spectral quality)
    - Multi-lead consensus option
    """

    def __init__(self, thresholds: OptimizedThresholds | None = None):
        self.thresholds = thresholds or OptimizedThresholds()
        self.is_tuned = False

    def tune(self, dataset, n_train: int = 400, verbose: bool = True):
        """
        Tune thresholds on training data.

        Args:
            dataset: ECGDataset
            n_train: Number of training samples
            verbose: Print progress
        """
        if verbose:
            print("Extracting features for training...")

        features_list = []
        labels = []

        # Balance classes
        n_per_class = n_train // 2

        for i, rid in enumerate(dataset.acceptable[:n_per_class]):
            try:
                ecg = dataset[rid]
                feat = extract_enhanced_features(ecg.signal, ecg.fs)
                features_list.append(feat)
                labels.append('Acceptable')
            except:
                pass
            if verbose and (i+1) % 50 == 0:
                print(f"  Acceptable: {i+1}/{n_per_class}")

        for i, rid in enumerate(dataset.unacceptable[:n_per_class]):
            try:
                ecg = dataset[rid]
                feat = extract_enhanced_features(ecg.signal, ecg.fs)
                features_list.append(feat)
                labels.append('Unacceptable')
            except:
                pass
            if verbose and (i+1) % 50 == 0:
                print(f"  Unacceptable: {i+1}/{n_per_class}")

        if verbose:
            print(f"\nRunning grid search on {len(features_list)} samples...")

        self.thresholds, self.search_results = grid_search_thresholds(
            features_list, labels, verbose=verbose
        )
        self.is_tuned = True

    def classify(self, signal: np.ndarray, fs: float, use_multilead: bool = False) -> dict:
        """
        Classify ECG quality.

        Args:
            signal: ECG signal (1D or 2D)
            fs: Sampling frequency
            use_multilead: Use multi-lead consensus

        Returns:
            Classification result dictionary
        """
        if use_multilead and signal.ndim == 2 and signal.shape[1] >= 2:
            return multi_lead_quality(signal, fs, self.thresholds)

        # Single-lead or single analysis
        features = extract_enhanced_features(signal, fs)
        prediction, reasons = classify_with_thresholds(features, self.thresholds)

        return {
            'prediction': prediction,
            'reasons': reasons,
            'features': features
        }

    def evaluate(self, dataset, record_ids: list, use_multilead: bool = False, verbose: bool = True) -> dict:
        """
        Evaluate on a set of records.

        Args:
            dataset: ECGDataset
            record_ids: List of record IDs to evaluate
            use_multilead: Use multi-lead consensus
            verbose: Print progress

        Returns:
            Metrics dictionary
        """
        predictions = []
        labels = []

        for i, rid in enumerate(record_ids):
            try:
                ecg = dataset[rid]
                result = self.classify(ecg.signal, ecg.fs, use_multilead)
                predictions.append(result['prediction'])
                labels.append(ecg.quality)
            except:
                pass

            if verbose and (i+1) % 50 == 0:
                print(f"  Evaluated {i+1}/{len(record_ids)}")

        # Calculate metrics
        pred_binary = np.array([1 if p == 'Acceptable' else 0 for p in predictions])
        label_binary = np.array([1 if l == 'Acceptable' else 0 for l in labels])

        accuracy = accuracy_score(label_binary, pred_binary)
        precision = precision_score(label_binary, pred_binary, zero_division=0)
        recall = recall_score(label_binary, pred_binary, zero_division=0)
        f1 = f1_score(label_binary, pred_binary, zero_division=0)

        cm = confusion_matrix(label_binary, pred_binary)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)},
            'n_samples': len(predictions)
        }
