"""
ECG Quality Classifier

Rule-based heuristic classification following the Moody 2011 approach.
Uses interpretable thresholds on extracted features.

Advantages:
- Interpretable: Explains WHY an ECG was rejected
- No training required (but thresholds can be tuned)
- Fast execution (suitable for mobile apps)
- Based on physiological and signal quality reasoning
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from .features import extract_all_features, ECGFeatures


@dataclass
class ClassificationResult:
    """Result of ECG quality classification."""
    prediction: Literal['Acceptable', 'Unacceptable']
    confidence: float                    # 0-1 confidence score
    reasons: list[str]                   # List of rejection reasons (if unacceptable)
    feature_flags: dict[str, bool]       # Which features passed/failed


@dataclass
class QualityThresholds:
    """
    Configurable thresholds for quality classification.

    Each threshold defines the boundary between acceptable and unacceptable.
    Values are based on physiological limits and signal quality standards.
    """
    # SNR thresholds
    snr_min: float = 5.0                 # Minimum SNR in dB

    # Beat detection thresholds (for 10s recording)
    beat_count_min: int = 5              # Minimum beats (30 bpm)
    beat_count_max: int = 30             # Maximum beats (180 bpm)

    # Heart rate thresholds
    hr_min: float = 30.0                 # Minimum HR (bpm)
    hr_max: float = 200.0                # Maximum HR (bpm)

    # RR variability threshold
    rr_cv_max: float = 0.5               # Maximum coefficient of variation

    # Baseline wander threshold
    baseline_power_ratio_max: float = 0.3  # Maximum baseline/signal power ratio

    # Spectral thresholds
    cardiac_power_ratio_min: float = 0.5   # Minimum power in cardiac band
    high_freq_ratio_max: float = 0.3       # Maximum high-frequency power

    # Time-frequency thresholds
    artifact_frame_ratio_max: float = 0.2  # Maximum artifact frames
    stationarity_min: float = 0.3          # Minimum stationarity index

    # Lead correlation threshold
    mean_correlation_min: float = 0.3      # Minimum inter-lead correlation

    # Spectral shape thresholds (from psdpar.m)
    spectral_flatness_max: float = 0.4     # Maximum flatness (>0.5 = noise-like)
    spectral_spread_max: float = 25.0      # Maximum f_std (Hz)
    spectral_kurtosis_min: float = 1.0     # Minimum kurtosis (peakiness)

    # Frequency distribution thresholds (from psdpar.m analysis)
    f_mean_min: float = 8.5                # Minimum spectral centroid (Hz) - catches drift
    f_std_min: float = 11.0                # Minimum spectral spread (Hz) - catches narrow spectrum
    f_iqr_min: float = 7.0                 # Minimum IQR (Hz) - catches limited frequency range


class QualityClassifier:
    """
    Rule-based ECG quality classifier.

    Uses a set of interpretable rules based on signal features.
    Each rule corresponds to a specific quality issue.

    Usage:
        classifier = QualityClassifier()
        result = classifier.classify(ecg_signal, fs)

        if result.prediction == 'Unacceptable':
            print("Rejection reasons:", result.reasons)
    """

    def __init__(self, thresholds: QualityThresholds | None = None):
        """
        Initialize classifier with thresholds.

        Args:
            thresholds: Custom thresholds or None for defaults
        """
        self.thresholds = thresholds or QualityThresholds()

    def classify_from_features(self, features: ECGFeatures) -> ClassificationResult:
        """
        Classify ECG quality from pre-extracted features.

        Args:
            features: ECGFeatures object from extract_all_features()

        Returns:
            ClassificationResult with prediction, confidence, and reasons
        """
        reasons = []
        feature_flags = {}
        th = self.thresholds

        # ===========================================
        # Rule 1: SNR Check
        # ===========================================
        snr_db = features.snr.get('snr_db', 0)
        snr_ok = snr_db >= th.snr_min and not np.isinf(snr_db) and not np.isnan(snr_db)
        feature_flags['snr'] = snr_ok

        if not snr_ok:
            if np.isinf(snr_db) or np.isnan(snr_db) or snr_db < 0:
                reasons.append(f"Very low SNR (signal dominated by noise)")
            else:
                reasons.append(f"Low SNR ({snr_db:.1f} dB < {th.snr_min} dB)")

        # ===========================================
        # Rule 2: Beat Count Check
        # ===========================================
        beat_count = features.morphological.get('beat_count', 0)
        beats_ok = th.beat_count_min <= beat_count <= th.beat_count_max
        feature_flags['beat_count'] = beats_ok

        if not beats_ok:
            if beat_count < th.beat_count_min:
                reasons.append(f"Too few beats detected ({beat_count} < {th.beat_count_min})")
            else:
                reasons.append(f"Too many beats detected ({beat_count} > {th.beat_count_max}, possible noise)")

        # ===========================================
        # Rule 3: Heart Rate Check
        # ===========================================
        hr = features.morphological.get('heart_rate', 0)
        hr_ok = th.hr_min <= hr <= th.hr_max if hr > 0 else False
        feature_flags['heart_rate'] = hr_ok

        if not hr_ok and beat_count >= th.beat_count_min:
            if hr < th.hr_min:
                reasons.append(f"Heart rate too low ({hr:.0f} bpm)")
            elif hr > th.hr_max:
                reasons.append(f"Heart rate too high ({hr:.0f} bpm)")

        # ===========================================
        # Rule 4: RR Regularity Check
        # ===========================================
        rr_cv = features.morphological.get('rr_cv', 0)
        # Need at least 3 beats to assess rhythm regularity; otherwise fail
        rr_ok = rr_cv <= th.rr_cv_max if beat_count >= 3 else False
        feature_flags['rr_regularity'] = rr_ok

        if not rr_ok:
            if beat_count < 3:
                reasons.append(f"Cannot assess rhythm (only {beat_count} beats)")
            else:
                reasons.append(f"Irregular rhythm (CV={rr_cv:.2f} > {th.rr_cv_max})")

        # ===========================================
        # Rule 5: Baseline Wander Check
        # ===========================================
        baseline_ratio = features.baseline.get('baseline_power_ratio', 0)
        baseline_ok = baseline_ratio <= th.baseline_power_ratio_max
        feature_flags['baseline'] = baseline_ok

        if not baseline_ok:
            reasons.append(f"Excessive baseline wander (ratio={baseline_ratio:.2f})")

        # ===========================================
        # Rule 6: Cardiac Power Check
        # ===========================================
        cardiac_ratio = features.spectral.get('cardiac_power_ratio', 0)
        cardiac_ok = cardiac_ratio >= th.cardiac_power_ratio_min
        feature_flags['cardiac_power'] = cardiac_ok

        if not cardiac_ok:
            reasons.append(f"Low cardiac frequency content ({cardiac_ratio:.1%} < {th.cardiac_power_ratio_min:.0%})")

        # ===========================================
        # Rule 7: High Frequency Noise Check
        # ===========================================
        hf_ratio = features.spectral.get('high_freq_power_ratio', 0)
        hf_ok = hf_ratio <= th.high_freq_ratio_max
        feature_flags['high_freq_noise'] = hf_ok

        if not hf_ok:
            reasons.append(f"High-frequency noise detected ({hf_ratio:.1%} > {th.high_freq_ratio_max:.0%})")

        # ===========================================
        # Rule 8: Artifact Detection (Time-Frequency)
        # ===========================================
        artifact_ratio = features.timefreq.get('artifact_frame_ratio', 0)
        artifact_ok = artifact_ratio <= th.artifact_frame_ratio_max
        feature_flags['artifacts'] = artifact_ok

        if not artifact_ok:
            reasons.append(f"Artifacts detected ({artifact_ratio:.0%} of frames)")

        # ===========================================
        # Rule 9: Stationarity Check
        # ===========================================
        stationarity = features.timefreq.get('stationarity_index', 0)
        stationary_ok = stationarity >= th.stationarity_min
        feature_flags['stationarity'] = stationary_ok

        if not stationary_ok:
            reasons.append(f"Non-stationary signal (index={stationarity:.2f})")

        # ===========================================
        # Rule 10: Lead Correlation (if multi-lead)
        # ===========================================
        mean_corr = features.lead_correlation.get('mean_correlation', 1.0)
        corr_ok = mean_corr >= th.mean_correlation_min
        feature_flags['lead_correlation'] = corr_ok

        if not corr_ok:
            reasons.append(f"Low inter-lead correlation ({mean_corr:.2f})")

        # ===========================================
        # Rule 11: Spectral Flatness (psdpar.m)
        # Good ECG: Low flatness (periodic QRS pattern)
        # Bad ECG: High flatness (noise-like, white noise = 1)
        # ===========================================
        spectral_flatness = features.spectral.get('spectral_flatness', 0)
        flatness_ok = spectral_flatness <= th.spectral_flatness_max
        feature_flags['spectral_flatness'] = flatness_ok

        if not flatness_ok:
            reasons.append(f"High spectral flatness ({spectral_flatness:.2f} > {th.spectral_flatness_max}, noise-like)")

        # ===========================================
        # Rule 12: Spectral Spread (psdpar.m)
        # Good ECG: Narrow spread (power concentrated in cardiac band)
        # Bad ECG: Wide spread (power distributed across frequencies)
        # ===========================================
        f_std = features.spectral.get('f_std', 0)
        spread_ok = f_std <= th.spectral_spread_max
        feature_flags['spectral_spread'] = spread_ok

        if not spread_ok:
            reasons.append(f"High spectral spread ({f_std:.1f} Hz > {th.spectral_spread_max} Hz)")

        # ===========================================
        # Rule 13: Spectral Kurtosis (psdpar.m)
        # Good ECG: Positive kurtosis (sharp spectral peaks)
        # Bad ECG: Low/zero kurtosis (flat spectrum, Gaussian-like)
        # ===========================================
        spectral_kurtosis = features.spectral.get('spectral_kurtosis', 0)
        kurtosis_ok = spectral_kurtosis >= th.spectral_kurtosis_min
        feature_flags['spectral_kurtosis'] = kurtosis_ok

        if not kurtosis_ok:
            reasons.append(f"Low spectral kurtosis ({spectral_kurtosis:.1f} < {th.spectral_kurtosis_min}, flat spectrum)")

        # ===========================================
        # Rule 14: Spectral Centroid (f_mean from psdpar.m)
        # Good ECG: f_mean ~10 Hz (cardiac content 5-15 Hz)
        # Bad ECG: f_mean < 8.5 Hz (drift-dominated, low freq content)
        # ===========================================
        f_mean = features.spectral.get('f_mean', 0)
        f_mean_ok = f_mean >= th.f_mean_min
        feature_flags['f_mean'] = f_mean_ok

        if not f_mean_ok:
            reasons.append(f"Low spectral centroid ({f_mean:.1f} Hz < {th.f_mean_min} Hz, drift-dominated)")

        # ===========================================
        # Rule 15: Minimum Spectral Spread (f_std from psdpar.m)
        # Good ECG: f_std ~14 Hz (wide spectrum with cardiac harmonics)
        # Bad ECG: f_std < 11 Hz (narrow spectrum, limited content)
        # ===========================================
        f_std_val = features.spectral.get('f_std', 0)
        f_std_min_ok = f_std_val >= th.f_std_min
        feature_flags['f_std_min'] = f_std_min_ok

        if not f_std_min_ok:
            reasons.append(f"Narrow frequency spectrum ({f_std_val:.1f} Hz < {th.f_std_min} Hz)")

        # ===========================================
        # Rule 16: Interquartile Range (f_iqr from psdpar.m)
        # Good ECG: f_iqr ~9 Hz (power spread across cardiac band)
        # Bad ECG: f_iqr < 7 Hz (concentrated power, poor quality)
        # ===========================================
        f_iqr = features.spectral.get('f_iqr', 0)
        f_iqr_ok = f_iqr >= th.f_iqr_min
        feature_flags['f_iqr'] = f_iqr_ok

        if not f_iqr_ok:
            reasons.append(f"Limited frequency range (IQR={f_iqr:.1f} Hz < {th.f_iqr_min} Hz)")

        # ===========================================
        # Final Decision
        # ===========================================
        n_passed = sum(feature_flags.values())
        n_total = len(feature_flags)

        # Calculate confidence based on how many rules passed
        confidence = n_passed / n_total

        # Decision: Acceptable if no critical failures
        # Critical: SNR, beat count, cardiac power
        critical_ok = feature_flags['snr'] and feature_flags['beat_count'] and feature_flags['cardiac_power']

        # Also check if majority of rules passed
        majority_ok = n_passed >= n_total * 0.7  # 70% of rules must pass

        if critical_ok and majority_ok:
            prediction = 'Acceptable'
        else:
            prediction = 'Unacceptable'

        return ClassificationResult(
            prediction=prediction,
            confidence=confidence,
            reasons=reasons,
            feature_flags=feature_flags
        )

    def classify(
        self,
        signal: np.ndarray,
        fs: float,
        r_peaks: np.ndarray | None = None
    ) -> ClassificationResult:
        """
        Classify ECG quality from raw signal.

        Args:
            signal: ECG signal (1D or 2D for multi-lead)
            fs: Sampling frequency (Hz)
            r_peaks: Optional pre-detected R-peaks

        Returns:
            ClassificationResult
        """
        features = extract_all_features(signal, fs, r_peaks)
        return self.classify_from_features(features)

    def evaluate(
        self,
        dataset,
        max_records: int | None = None,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate classifier on a dataset.

        Args:
            dataset: ECGDataset instance
            max_records: Maximum records to evaluate
            verbose: Print progress

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = []
        labels = []
        all_reasons = []

        n_records = min(len(dataset), max_records) if max_records else len(dataset)

        for i in range(n_records):
            try:
                ecg = dataset[i]
                result = self.classify(ecg.signal, ecg.fs)

                predictions.append(result.prediction)
                labels.append(ecg.quality)
                all_reasons.append(result.reasons)

                if verbose and (i + 1) % 100 == 0:
                    print(f"  Evaluated {i + 1}/{n_records}...")

            except Exception as e:
                if verbose:
                    print(f"  Error on record {i}: {e}")
                continue

        # Calculate metrics
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Confusion matrix elements
        tp = np.sum((predictions == 'Acceptable') & (labels == 'Acceptable'))
        tn = np.sum((predictions == 'Unacceptable') & (labels == 'Unacceptable'))
        fp = np.sum((predictions == 'Acceptable') & (labels == 'Unacceptable'))
        fn = np.sum((predictions == 'Unacceptable') & (labels == 'Acceptable'))

        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn),
                'fp': int(fp), 'fn': int(fn)
            },
            'n_evaluated': len(predictions),
            'predictions': predictions,
            'labels': labels,
            'reasons': all_reasons
        }



def cross_validate(
    dataset,
    n_folds: int = 5,
    verbose: bool = True
) -> dict:
    """
    Perform k-fold cross-validation with threshold learning.

    For each fold:
    - Train: Learn optimal thresholds from training data
    - Test: Evaluate on held-out test data

    Args:
        dataset: ECGDataset instance
        n_folds: Number of folds (default: 5)
        verbose: Print progress

    Returns:
        Dictionary with average metrics and per-fold results
    """
    from .features import extract_all_features

    # Get all record IDs by class for stratified splitting
    acceptable_ids = list(dataset.acceptable)
    unacceptable_ids = list(dataset.unacceptable)

    # Shuffle for randomness
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(acceptable_ids)
    np.random.shuffle(unacceptable_ids)

    # Create stratified folds
    def split_into_folds(ids, n_folds):
        fold_size = len(ids) // n_folds
        folds = []
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else len(ids)
            folds.append(ids[start:end])
        return folds

    acceptable_folds = split_into_folds(acceptable_ids, n_folds)
    unacceptable_folds = split_into_folds(unacceptable_ids, n_folds)

    fold_results = []

    for fold_idx in range(n_folds):
        if verbose:
            print(f"Fold {fold_idx + 1}/{n_folds}")

        # Split into train/test
        test_acceptable = acceptable_folds[fold_idx]
        test_unacceptable = unacceptable_folds[fold_idx]

        train_acceptable = []
        train_unacceptable = []
        for i in range(n_folds):
            if i != fold_idx:
                train_acceptable.extend(acceptable_folds[i])
                train_unacceptable.extend(unacceptable_folds[i])

        # Extract features from training data
        train_good_features = []
        train_bad_features = []

        for record_id in train_acceptable:
            try:
                ecg = dataset[record_id]
                features = extract_all_features(ecg.signal, ecg.fs)
                train_good_features.append(features.to_dict())
            except:
                pass

        for record_id in train_unacceptable:
            try:
                ecg = dataset[record_id]
                features = extract_all_features(ecg.signal, ecg.fs)
                train_bad_features.append(features.to_dict())
            except:
                pass

        # Learn thresholds from training data
        thresholds = _learn_thresholds(train_good_features, train_bad_features)

        # Evaluate on test data
        classifier = QualityClassifier(thresholds)

        test_ids = test_acceptable + test_unacceptable
        predictions = []
        labels = []

        for record_id in test_ids:
            try:
                ecg = dataset[record_id]
                result = classifier.classify(ecg.signal, ecg.fs)
                predictions.append(result.prediction)
                labels.append(ecg.quality)
            except:
                pass

        # Calculate metrics for this fold
        predictions = np.array(predictions)
        labels = np.array(labels)

        tp = np.sum((predictions == 'Acceptable') & (labels == 'Acceptable'))
        tn = np.sum((predictions == 'Unacceptable') & (labels == 'Unacceptable'))
        fp = np.sum((predictions == 'Acceptable') & (labels == 'Unacceptable'))
        fn = np.sum((predictions == 'Unacceptable') & (labels == 'Acceptable'))

        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        fold_result = {
            'fold': fold_idx + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity
        }
        fold_results.append(fold_result)

        if verbose:
            print(f"  Accuracy: {accuracy*100:.1f}%, F1: {f1*100:.1f}%, Specificity: {specificity*100:.1f}%")

    # Calculate average metrics
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])
    avg_f1 = np.mean([r['f1_score'] for r in fold_results])
    avg_specificity = np.mean([r['specificity'] for r in fold_results])

    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    std_f1 = np.std([r['f1_score'] for r in fold_results])

    if verbose:
        print(f"\n{'='*50}")
        print(f"Cross-Validation Results ({n_folds}-fold)")
        print(f"{'='*50}")
        print(f"Accuracy:    {avg_accuracy*100:.1f}% +/- {std_accuracy*100:.1f}%")
        print(f"Precision:   {avg_precision*100:.1f}%")
        print(f"Recall:      {avg_recall*100:.1f}%")
        print(f"Specificity: {avg_specificity*100:.1f}%")
        print(f"F1 Score:    {avg_f1*100:.1f}% +/- {std_f1*100:.1f}%")

    return {
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1_score': avg_f1,
        'avg_specificity': avg_specificity,
        'std_accuracy': std_accuracy,
        'std_f1_score': std_f1,
        'fold_results': fold_results,
        'n_folds': n_folds
    }


def _learn_thresholds(good_features: list, bad_features: list) -> QualityThresholds:
    """
    Learn optimal thresholds from labeled training data.
    """
    def get_values(features_list, key):
        vals = [f.get(key) for f in features_list if f.get(key) is not None]
        vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
        return vals

    def find_threshold(good_vals, bad_vals, higher_is_better=True):
        if not good_vals or not bad_vals:
            return None
        if higher_is_better:
            good_p25 = np.percentile(good_vals, 25)
            bad_p75 = np.percentile(bad_vals, 75)
            return (good_p25 + bad_p75) / 2
        else:
            good_p75 = np.percentile(good_vals, 75)
            bad_p25 = np.percentile(bad_vals, 25)
            return (good_p75 + bad_p25) / 2

    thresholds = QualityThresholds()

    # SNR (higher is better)
    snr_good = get_values(good_features, 'snr_snr_db')
    snr_bad = get_values(bad_features, 'snr_snr_db')
    if snr_good and snr_bad:
        val = find_threshold(snr_good, snr_bad, True)
        if val is not None:
            thresholds.snr_min = max(0, val)

    # RR CV (lower is better)
    cv_good = get_values(good_features, 'morph_rr_cv')
    cv_bad = get_values(bad_features, 'morph_rr_cv')
    if cv_good and cv_bad:
        val = find_threshold(cv_good, cv_bad, False)
        if val is not None:
            thresholds.rr_cv_max = val

    # Cardiac power ratio (higher is better)
    cardiac_good = get_values(good_features, 'spectral_cardiac_power_ratio')
    cardiac_bad = get_values(bad_features, 'spectral_cardiac_power_ratio')
    if cardiac_good and cardiac_bad:
        val = find_threshold(cardiac_good, cardiac_bad, True)
        if val is not None:
            thresholds.cardiac_power_ratio_min = val

    # Stationarity (higher is better)
    stat_good = get_values(good_features, 'tf_stationarity_index')
    stat_bad = get_values(bad_features, 'tf_stationarity_index')
    if stat_good and stat_bad:
        val = find_threshold(stat_good, stat_bad, True)
        if val is not None:
            thresholds.stationarity_min = val

    # f_mean (higher is better)
    fmean_good = get_values(good_features, 'spectral_f_mean')
    fmean_bad = get_values(bad_features, 'spectral_f_mean')
    if fmean_good and fmean_bad:
        val = find_threshold(fmean_good, fmean_bad, True)
        if val is not None:
            thresholds.f_mean_min = val

    # f_std (higher is better)
    fstd_good = get_values(good_features, 'spectral_f_std')
    fstd_bad = get_values(bad_features, 'spectral_f_std')
    if fstd_good and fstd_bad:
        val = find_threshold(fstd_good, fstd_bad, True)
        if val is not None:
            thresholds.f_std_min = val

    # f_iqr (higher is better)
    fiqr_good = get_values(good_features, 'spectral_f_iqr')
    fiqr_bad = get_values(bad_features, 'spectral_f_iqr')
    if fiqr_good and fiqr_bad:
        val = find_threshold(fiqr_good, fiqr_bad, True)
        if val is not None:
            thresholds.f_iqr_min = val

    return thresholds


def tune_thresholds(
    dataset,
    n_samples: int = 200,
    verbose: bool = True
) -> QualityThresholds:
    """
    Tune thresholds based on dataset statistics.

    Uses the distribution of features in acceptable vs unacceptable
    ECGs to set optimal thresholds.

    Args:
        dataset: ECGDataset instance
        n_samples: Number of samples per class
        verbose: Print progress

    Returns:
        Tuned QualityThresholds
    """
    from .features import extract_all_features

    acceptable_features = []
    unacceptable_features = []

    if verbose:
        print("Extracting features for threshold tuning...")

    # Extract features from acceptable ECGs
    for i, record_id in enumerate(dataset.acceptable[:n_samples]):
        try:
            ecg = dataset[record_id]
            features = extract_all_features(ecg.signal, ecg.fs)
            acceptable_features.append(features.to_dict())
        except:
            pass
        if verbose and (i + 1) % 50 == 0:
            print(f"  Acceptable: {i + 1}/{min(n_samples, len(dataset.acceptable))}")

    # Extract features from unacceptable ECGs
    for i, record_id in enumerate(dataset.unacceptable[:n_samples]):
        try:
            ecg = dataset[record_id]
            features = extract_all_features(ecg.signal, ecg.fs)
            unacceptable_features.append(features.to_dict())
        except:
            pass
        if verbose and (i + 1) % 50 == 0:
            print(f"  Unacceptable: {i + 1}/{min(n_samples, len(dataset.unacceptable))}")

    def get_values(features_list, key):
        vals = [f.get(key) for f in features_list if f.get(key) is not None]
        vals = [v for v in vals if not np.isnan(v) and not np.isinf(v)]
        return vals

    def find_threshold(good_vals, bad_vals, higher_is_better=True):
        """Find threshold that best separates good from bad."""
        if not good_vals or not bad_vals:
            return None

        if higher_is_better:
            # Threshold should be below good values, above bad values
            good_p25 = np.percentile(good_vals, 25)
            bad_p75 = np.percentile(bad_vals, 75)
            return (good_p25 + bad_p75) / 2
        else:
            # Threshold should be above good values, below bad values
            good_p75 = np.percentile(good_vals, 75)
            bad_p25 = np.percentile(bad_vals, 25)
            return (good_p75 + bad_p25) / 2

    # Tune each threshold
    thresholds = QualityThresholds()

    # SNR (higher is better)
    snr_good = get_values(acceptable_features, 'snr_snr_db')
    snr_bad = get_values(unacceptable_features, 'snr_snr_db')
    if snr_good and snr_bad:
        thresholds.snr_min = max(0, find_threshold(snr_good, snr_bad, True))

    # RR CV (lower is better)
    cv_good = get_values(acceptable_features, 'morph_rr_cv')
    cv_bad = get_values(unacceptable_features, 'morph_rr_cv')
    if cv_good and cv_bad:
        thresholds.rr_cv_max = find_threshold(cv_good, cv_bad, False)

    # Cardiac power ratio (higher is better)
    cardiac_good = get_values(acceptable_features, 'spectral_cardiac_power_ratio')
    cardiac_bad = get_values(unacceptable_features, 'spectral_cardiac_power_ratio')
    if cardiac_good and cardiac_bad:
        thresholds.cardiac_power_ratio_min = find_threshold(cardiac_good, cardiac_bad, True)

    # Stationarity (higher is better)
    stat_good = get_values(acceptable_features, 'tf_stationarity_index')
    stat_bad = get_values(unacceptable_features, 'tf_stationarity_index')
    if stat_good and stat_bad:
        thresholds.stationarity_min = find_threshold(stat_good, stat_bad, True)

    if verbose:
        print(f"\nTuned thresholds:")
        print(f"  SNR min: {thresholds.snr_min:.1f} dB")
        print(f"  RR CV max: {thresholds.rr_cv_max:.2f}")
        print(f"  Cardiac power min: {thresholds.cardiac_power_ratio_min:.2f}")
        print(f"  Stationarity min: {thresholds.stationarity_min:.2f}")

    return thresholds
