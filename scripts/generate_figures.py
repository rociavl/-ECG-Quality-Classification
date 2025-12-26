"""
Generate Presentation Figures for ECG Quality Assessment

This script generates all figures needed for the BSA course presentation.
Each figure demonstrates theory understanding and can be used directly in slides.

Figures:
1. Pan-Tompkins algorithm steps (good vs bad ECG)
2. Spectral analysis comparison
3. Time-frequency spectrogram
4. Filtering demo
5. Classification results
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, spectrogram, periodogram

from src.data_loader import ECGDataset
from src.pan_tompkins import pan_tompkins
from src.spectral_params import compute_psd_parameters
from src.ar_order import select_ar_order


# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def get_good_bad_ecg(dataset: ECGDataset) -> tuple:
    """
    Find representative good and bad ECG examples from dataset.

    Returns:
        (good_ecg, bad_ecg) tuple with ECGRecord objects
    """
    # Use dataset properties for acceptable/unacceptable lists
    acceptable = dataset.acceptable
    unacceptable = dataset.unacceptable

    print(f"  Found {len(acceptable)} acceptable, {len(unacceptable)} unacceptable records")

    # Find good ECG (clear signal, detectable beats)
    good_ecg = None
    for record_id in acceptable[:50]:  # Check first 50
        try:
            ecg = dataset[record_id]
            # Check if Pan-Tompkins finds reasonable number of peaks
            result = pan_tompkins(ecg.signal[:, 1], ecg.fs)
            if 8 <= len(result.qrs_peaks) <= 15:  # 48-90 bpm
                good_ecg = ecg
                print(f"  Selected good ECG: {record_id} ({len(result.qrs_peaks)} peaks)")
                break
        except Exception as e:
            continue

    # Find bad ECG (visible artifacts/noise)
    bad_ecg = None
    for record_id in unacceptable[:50]:
        try:
            ecg = dataset[record_id]
            bad_ecg = ecg
            print(f"  Selected bad ECG: {record_id}")
            break
        except Exception as e:
            continue

    return good_ecg, bad_ecg


def visualize_pan_tompkins_theory(good_ecg, bad_ecg, save_path: Path):
    """
    Show Pan-Tompkins algorithm steps on good vs bad ECG.

    THIS is what professors want to see - step-by-step algorithm demonstration!

    6 rows × 2 columns:
        Row 0: Original signal
        Row 1: Bandpass filtered (5-15 Hz)
        Row 2: Derivative (highlights slopes)
        Row 3: Squared (emphasizes peaks)
        Row 4: Moving window integrated
        Row 5: Final detected R-peaks on original
    """
    fig, axes = plt.subplots(6, 2, figsize=(14, 14))

    step_names = [
        'Step 0: Original ECG',
        'Step 1: Bandpass Filter (5-15 Hz)',
        'Step 2: Derivative (slope detection)',
        'Step 3: Squaring (emphasize peaks)',
        'Step 4: Moving Window Integration',
        'Step 5: R-peak Detection'
    ]

    for col, (ecg, title) in enumerate([(good_ecg, 'GOOD ECG (Acceptable)'),
                                         (bad_ecg, 'BAD ECG (Unacceptable)')]):
        signal = ecg.signal[:, 1]  # Lead II
        fs = ecg.fs
        t = np.arange(len(signal)) / fs

        # Run Pan-Tompkins step by step
        # Step 1: Bandpass filter
        nyq = fs / 2
        b, a = butter(3, [5/nyq, 15/nyq], btype='band')
        filtered = filtfilt(b, a, signal)
        filtered = filtered / (np.max(np.abs(filtered)) + 1e-10)

        # Step 2: Derivative
        h_d = np.array([-1, -2, 0, 2, 1]) / 8
        derivative = np.convolve(filtered, h_d, mode='same')
        derivative = derivative / (np.max(np.abs(derivative)) + 1e-10)

        # Step 3: Squaring
        squared = derivative ** 2

        # Step 4: Moving window integration
        window_width = int(0.150 * fs)  # 150ms
        mwi_kernel = np.ones(window_width) / window_width
        integrated = np.convolve(squared, mwi_kernel, mode='same')

        # Step 5: Peak detection (using full Pan-Tompkins)
        result = pan_tompkins(signal, fs)
        peaks = result.qrs_peaks

        # Plot each step
        signals = [signal, filtered, derivative, squared, integrated, signal]

        for row in range(6):
            ax = axes[row, col]

            if row == 5:
                # Final step: show peaks on original
                ax.plot(t, signals[row], 'b-', linewidth=0.8)
                if len(peaks) > 0:
                    ax.plot(t[peaks], signal[peaks], 'ro', markersize=8, label=f'{len(peaks)} R-peaks')
                    ax.legend(loc='upper right')
            else:
                ax.plot(t, signals[row], 'b-', linewidth=0.8)

            ax.set_xlim([0, 10])
            ax.set_ylabel('Amplitude')

            if row == 0:
                ax.set_title(title, fontsize=12, fontweight='bold')

            if col == 0:
                ax.annotate(step_names[row], xy=(0, 0.5), xytext=(-0.15, 0.5),
                           xycoords='axes fraction', textcoords='axes fraction',
                           fontsize=9, rotation=90, va='center', ha='right',
                           fontweight='bold')

            if row == 5:
                ax.set_xlabel('Time (s)')

    plt.suptitle('Pan-Tompkins QRS Detection Algorithm: Step-by-Step Processing',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {save_path}")


def visualize_spectral_analysis(good_ecg, bad_ecg, save_path: Path):
    """
    Show PSD comparison between good and bad ECG.

    Demonstrates frequency-domain analysis from BSA course.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row, (ecg, title) in enumerate([(good_ecg, 'Good ECG'),
                                         (bad_ecg, 'Bad ECG')]):
        signal = ecg.signal[:, 1]  # Lead II
        fs = ecg.fs
        t = np.arange(len(signal)) / fs

        # Time domain
        ax = axes[row, 0]
        ax.plot(t, signal, 'b-', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (mV)')
        ax.set_title(f'{title} - Time Domain')
        ax.set_xlim([0, 10])

        # Frequency domain (PSD)
        ax = axes[row, 1]
        f, Pxx = periodogram(signal, fs, window='hamming')

        # Plot up to 50 Hz
        mask = f <= 50
        ax.semilogy(f[mask], Pxx[mask], 'b-', linewidth=0.8)

        # Mark cardiac frequency range
        ax.axvspan(0.5, 2.5, alpha=0.2, color='green', label='Cardiac (0.5-2.5 Hz)')
        ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='Powerline (50 Hz)')

        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density')
        ax.set_title(f'{title} - Power Spectrum')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim([0, 50])

    plt.suptitle('Spectral Analysis: Good vs Bad ECG Quality',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {save_path}")


def visualize_timefreq_analysis(good_ecg, bad_ecg, save_path: Path):
    """
    Show STFT spectrogram comparison - BSA course requirement!

    Time-frequency representation shows how spectral content changes over time.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for row, (ecg, title) in enumerate([(good_ecg, 'Good ECG'),
                                         (bad_ecg, 'Bad ECG')]):
        signal = ecg.signal[:, 1]
        fs = ecg.fs
        t = np.arange(len(signal)) / fs

        # Time domain
        ax = axes[row, 0]
        ax.plot(t, signal, 'b-', linewidth=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{title} - Signal')
        ax.set_xlim([0, 10])

        # Spectrogram (STFT)
        ax = axes[row, 1]

        # STFT parameters from BSA course
        nperseg = int(fs * 0.5)  # 500ms window
        noverlap = int(nperseg * 0.9)  # 90% overlap

        f, t_spec, Sxx = spectrogram(signal, fs, nperseg=nperseg,
                                      noverlap=noverlap, window='hamming')

        # Plot up to 30 Hz
        freq_mask = f <= 30

        im = ax.pcolormesh(t_spec, f[freq_mask], 10*np.log10(Sxx[freq_mask, :] + 1e-10),
                          shading='gouraud', cmap='viridis')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'{title} - Spectrogram (STFT)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dB)')

    plt.suptitle('Time-Frequency Analysis: Spectrogram Comparison',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {save_path}")


def visualize_filtering_demo(ecg, save_path: Path):
    """
    Demonstrate effect of different filters.

    Shows preprocessing knowledge.
    """
    signal = ecg.signal[:, 1]
    fs = ecg.fs
    t = np.arange(len(signal)) / fs

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Original
    ax = axes[0]
    ax.plot(t, signal, 'b-', linewidth=0.5)
    ax.set_title('Original ECG Signal', fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.set_xlim([0, 10])

    # After highpass (remove baseline wander)
    nyq = fs / 2
    b, a = butter(4, 0.5/nyq, btype='high')
    highpassed = filtfilt(b, a, signal)

    ax = axes[1]
    ax.plot(t, highpassed, 'g-', linewidth=0.5)
    ax.set_title('After High-pass Filter (0.5 Hz) - Removes Baseline Wander', fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.set_xlim([0, 10])

    # After notch filter (remove powerline)
    from scipy.signal import iirnotch
    b_notch, a_notch = iirnotch(50, 30, fs)
    notched = filtfilt(b_notch, a_notch, highpassed)

    ax = axes[2]
    ax.plot(t, notched, 'orange', linewidth=0.5)
    ax.set_title('After Notch Filter (50 Hz) - Removes Powerline Interference', fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.set_xlim([0, 10])

    # After bandpass (clinical ECG)
    b, a = butter(4, [0.5/nyq, 100/nyq], btype='band')
    bandpassed = filtfilt(b, a, signal)

    ax = axes[3]
    ax.plot(t, bandpassed, 'purple', linewidth=0.5)
    ax.set_title('After Bandpass Filter (0.5-100 Hz) - Clinical ECG Bandwidth', fontweight='bold')
    ax.set_ylabel('Amplitude')
    ax.set_xlabel('Time (s)')
    ax.set_xlim([0, 10])

    plt.suptitle('ECG Filtering Pipeline: Step-by-Step Preprocessing',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {save_path}")


def visualize_spectral_parameters(dataset: ECGDataset, save_path: Path, n_samples: int = 50):
    """
    Compare spectral parameters (psdpar.m) between good and bad ECG.

    Shows the discriminative power of the 10 new spectral features.
    """
    print("  Extracting spectral parameters from samples...")

    good_params = []
    bad_params = []

    # Extract from good ECGs
    for record_id in dataset.acceptable[:n_samples]:
        try:
            ecg = dataset[record_id]
            signal = ecg.signal[:, 1]  # Lead II
            f, Pxx = periodogram(signal, ecg.fs, window='hamming')
            params = compute_psd_parameters(f, Pxx)
            good_params.append(params)
        except:
            continue

    # Extract from bad ECGs
    for record_id in dataset.unacceptable[:n_samples]:
        try:
            ecg = dataset[record_id]
            signal = ecg.signal[:, 1]
            f, Pxx = periodogram(signal, ecg.fs, window='hamming')
            params = compute_psd_parameters(f, Pxx)
            bad_params.append(params)
        except:
            continue

    print(f"  Extracted {len(good_params)} good, {len(bad_params)} bad samples")

    if len(good_params) == 0 or len(bad_params) == 0:
        print("  ERROR: Not enough samples")
        return

    # Parameters to plot (most discriminative)
    params_to_plot = [
        ('spectral_flatness', 'Spectral Flatness\n(Wiener Entropy)'),
        ('f_std', 'Spectral Spread\nf_std (Hz)'),
        ('spectral_kurtosis', 'Spectral Kurtosis\n(Peakiness)'),
        ('f_iqr', 'Interquartile Range\nf_iqr (Hz)'),
        ('spectral_asymmetry', 'Spectral Asymmetry\n(Skewness)'),
        ('f_mean', 'Spectral Centroid\nf_mean (Hz)')
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (param_key, param_label) in enumerate(params_to_plot):
        ax = axes[idx]

        good_vals = [p[param_key] for p in good_params if param_key in p]
        bad_vals = [p[param_key] for p in bad_params if param_key in p]

        # Box plot
        bp = ax.boxplot([good_vals, bad_vals],
                       tick_labels=['Good ECG', 'Bad ECG'],
                       patch_artist=True)

        # Colors
        bp['boxes'][0].set_facecolor('lightgreen')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_ylabel(param_label)
        ax.set_title(param_key, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add mean values as text
        good_mean = np.mean(good_vals) if good_vals else 0
        bad_mean = np.mean(bad_vals) if bad_vals else 0
        ax.text(0.95, 0.95, f'μ Good: {good_mean:.2f}\nμ Bad: {bad_mean:.2f}',
               transform=ax.transAxes, va='top', ha='right',
               fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Spectral Parameters (psdpar.m): Good vs Bad ECG Comparison',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {save_path}")


def visualize_spectral_scatter(dataset: ECGDataset, save_path: Path, n_samples: int = 100):
    """
    Scatter plot of spectral_flatness vs spectral_kurtosis.

    Shows class separation in 2D feature space.
    """
    print("  Extracting features for scatter plot...")

    good_flat, good_kurt = [], []
    bad_flat, bad_kurt = [], []

    # Extract from good ECGs
    for record_id in dataset.acceptable[:n_samples]:
        try:
            ecg = dataset[record_id]
            signal = ecg.signal[:, 1]
            f, Pxx = periodogram(signal, ecg.fs, window='hamming')
            params = compute_psd_parameters(f, Pxx)
            good_flat.append(params['spectral_flatness'])
            good_kurt.append(params['spectral_kurtosis'])
        except:
            continue

    # Extract from bad ECGs
    for record_id in dataset.unacceptable[:n_samples]:
        try:
            ecg = dataset[record_id]
            signal = ecg.signal[:, 1]
            f, Pxx = periodogram(signal, ecg.fs, window='hamming')
            params = compute_psd_parameters(f, Pxx)
            bad_flat.append(params['spectral_flatness'])
            bad_kurt.append(params['spectral_kurtosis'])
        except:
            continue

    print(f"  Extracted {len(good_flat)} good, {len(bad_flat)} bad samples")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(good_flat, good_kurt, c='green', alpha=0.6, s=50,
               label=f'Good ECG (n={len(good_flat)})', edgecolors='darkgreen')
    ax.scatter(bad_flat, bad_kurt, c='red', alpha=0.6, s=50,
               label=f'Bad ECG (n={len(bad_flat)})', edgecolors='darkred')

    # Add threshold lines from classifier
    ax.axvline(x=0.4, color='orange', linestyle='--', linewidth=2,
               label='Flatness threshold (0.4)')
    ax.axhline(y=1.0, color='purple', linestyle='--', linewidth=2,
               label='Kurtosis threshold (1.0)')

    ax.set_xlabel('Spectral Flatness (Wiener Entropy)', fontsize=12)
    ax.set_ylabel('Spectral Kurtosis (Peakiness)', fontsize=12)
    ax.set_title('Spectral Shape Analysis: Good vs Bad ECG\n(psdpar.m parameters)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add interpretation text
    ax.text(0.02, 0.98,
            'Good ECG: Low flatness (periodic)\n          High kurtosis (sharp peaks)',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.text(0.98, 0.02,
            'Bad ECG: High flatness (noise-like)\n         Low kurtosis (flat spectrum)',
            transform=ax.transAxes, va='bottom', ha='right',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {save_path}")


def visualize_ar_order_selection(good_ecg, bad_ecg, save_path: Path):
    """
    Show AR model order selection curves (arord.m).

    Demonstrates automatic order selection using AICm criterion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (ecg, title, color) in enumerate([
        (good_ecg, 'Good ECG', 'green'),
        (bad_ecg, 'Bad ECG', 'red')
    ]):
        signal = ecg.signal[:, 1]  # Lead II
        ax = axes[idx]

        # Compute AR order selection for all three criteria
        criteria = ['fpe', 'aic', 'aicm']
        colors_crit = ['blue', 'orange', 'purple']

        for crit, crit_color in zip(criteria, colors_crit):
            try:
                optimal_order, scores = select_ar_order(signal, max_order=30, criterion=crit)
                orders = list(range(1, len(scores) + 1))

                # Normalize scores for comparison
                scores_norm = (np.array(scores) - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-10)

                ax.plot(orders, scores_norm, '-o', color=crit_color,
                       label=f'{crit.upper()} (opt={optimal_order})',
                       markersize=4, linewidth=1.5)

                # Mark optimal order
                ax.axvline(x=optimal_order, color=crit_color, linestyle='--', alpha=0.5)
            except Exception as e:
                print(f"  Warning: {crit} failed for {title}: {e}")
                continue

        ax.set_xlabel('Model Order (p)', fontsize=11)
        ax.set_ylabel('Normalized Score (lower = better)', fontsize=11)
        ax.set_title(f'{title}: AR Order Selection\n(arord.m criteria)',
                    fontsize=12, fontweight='bold', color=color)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 32])

    plt.suptitle('AR Model Order Selection: FPE, AIC, AICm Comparison',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved: {save_path}")


def main():
    """Generate all presentation figures."""

    # Setup paths
    data_path = project_root / 'data' / 'challenge2011' / 'set-a'
    figures_path = project_root / 'results' / 'figures'
    figures_path.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = ECGDataset(data_path)
    print(f"Loaded {len(dataset)} records")

    print("\nFinding good and bad ECG examples...")
    good_ecg, bad_ecg = get_good_bad_ecg(dataset)

    if good_ecg is None or bad_ecg is None:
        print("ERROR: Could not find suitable ECG examples")
        return

    print(f"Good ECG: {good_ecg.record_id}")
    print(f"Bad ECG: {bad_ecg.record_id}")

    print("\n" + "="*60)
    print("GENERATING PRESENTATION FIGURES")
    print("="*60)

    # Figure 1: Pan-Tompkins Theory
    print("\n[1/7] Pan-Tompkins Theory Visualization...")
    visualize_pan_tompkins_theory(
        good_ecg, bad_ecg,
        figures_path / 'pan_tompkins_theory.png'
    )

    # Figure 2: Spectral Analysis
    print("[2/7] Spectral Analysis Comparison...")
    visualize_spectral_analysis(
        good_ecg, bad_ecg,
        figures_path / 'spectral_analysis.png'
    )

    # Figure 3: Time-Frequency
    print("[3/7] Time-Frequency Analysis...")
    visualize_timefreq_analysis(
        good_ecg, bad_ecg,
        figures_path / 'timefreq_analysis.png'
    )

    # Figure 4: Filtering Demo
    print("[4/7] Filtering Demo...")
    visualize_filtering_demo(
        good_ecg,  # Use good ECG to show filtering steps clearly
        figures_path / 'filtering_demo.png'
    )

    # Figure 5: Spectral Parameters (psdpar.m) - NEW
    print("[5/7] Spectral Parameters Comparison (psdpar.m)...")
    visualize_spectral_parameters(
        dataset,
        figures_path / 'spectral_parameters_comparison.png'
    )

    # Figure 6: Spectral Shape Scatter - NEW
    print("[6/7] Spectral Shape Scatter Plot...")
    visualize_spectral_scatter(
        dataset,
        figures_path / 'spectral_shape_scatter.png'
    )

    # Figure 7: AR Order Selection (arord.m) - NEW
    print("[7/7] AR Order Selection (arord.m)...")
    visualize_ar_order_selection(
        good_ecg, bad_ecg,
        figures_path / 'ar_order_selection.png'
    )

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED!")
    print("="*60)
    print(f"\nOutput directory: {figures_path}")
    print("\nFiles created:")
    for f in sorted(figures_path.glob('*.png')):
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
