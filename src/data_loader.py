"""
ECG Data Loading Module
PhysioNet Challenge 2011 - ECG Quality Classification
"""

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import wfdb


@dataclass
class ECGRecord:
    """Container for a single ECG record."""
    record_id: str
    signal: np.ndarray      # Shape: (samples, leads) = (5000, 12)
    fs: int                 # Sampling frequency (500 Hz)
    leads: list[str]        # Lead names
    quality: str | None     # 'Acceptable' or 'Unacceptable'


def load_ecg_record(record_path: str | Path, quality: str | None = None) -> ECGRecord:
    """
    Load a single ECG record from WFDB format.

    Args:
        record_path: Path to record (without extension)
        quality: Optional quality label

    Returns:
        ECGRecord with signal data and metadata
    """
    record = wfdb.rdrecord(str(record_path))

    return ECGRecord(
        record_id=Path(record_path).name,
        signal=record.p_signal,
        fs=record.fs,
        leads=record.sig_name,
        quality=quality
    )


def load_labels(csv_path: str | Path) -> pd.DataFrame:
    """
    Load reference labels from CSV.

    Args:
        csv_path: Path to reference_labels.csv

    Returns:
        DataFrame with columns: record_id, quality
    """
    df = pd.read_csv(csv_path)
    df['record_id'] = df['record_id'].astype(str)
    return df


def get_records_by_quality(df: pd.DataFrame, quality: str) -> list[str]:
    """
    Filter record IDs by quality label.

    Args:
        df: Labels DataFrame
        quality: 'Acceptable' or 'Unacceptable'

    Returns:
        List of record IDs
    """
    return df[df['quality'] == quality]['record_id'].tolist()


class ECGDataset:
    """
    Dataset class for batch ECG loading.

    Usage:
        dataset = ECGDataset(data_dir)
        ecg = dataset[0]           # Load by index
        ecg = dataset['1002867']   # Load by record ID

        for ecg in dataset:        # Iterate all
            process(ecg)
    """

    def __init__(self, data_dir: str | Path):
        """
        Initialize dataset from directory.

        Args:
            data_dir: Path to set-a directory
        """
        self.data_dir = Path(data_dir)
        self.labels_path = self.data_dir / 'reference_labels.csv'

        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels not found: {self.labels_path}")

        self.labels_df = load_labels(self.labels_path)
        self.record_ids = self.labels_df['record_id'].tolist()

        # Create lookup dict for quality
        self._quality_map = dict(zip(
            self.labels_df['record_id'],
            self.labels_df['quality']
        ))

    def __len__(self) -> int:
        return len(self.record_ids)

    def __getitem__(self, key: int | str) -> ECGRecord:
        """Load ECG by index or record ID."""
        if isinstance(key, int):
            record_id = self.record_ids[key]
        else:
            record_id = str(key)

        record_path = self.data_dir / record_id
        quality = self._quality_map.get(record_id)

        return load_ecg_record(record_path, quality)

    def __iter__(self):
        """Iterate over all records."""
        for record_id in self.record_ids:
            yield self[record_id]

    @property
    def acceptable(self) -> list[str]:
        """List of acceptable record IDs."""
        return get_records_by_quality(self.labels_df, 'Acceptable')

    @property
    def unacceptable(self) -> list[str]:
        """List of unacceptable record IDs."""
        return get_records_by_quality(self.labels_df, 'Unacceptable')

    def summary(self) -> dict:
        """Return dataset statistics."""
        return {
            'total': len(self),
            'acceptable': len(self.acceptable),
            'unacceptable': len(self.unacceptable),
            'fs': 500,
            'duration_sec': 10,
            'n_leads': 12
        }


# Default data path
DEFAULT_DATA_DIR = Path(__file__).parent.parent / 'data' / 'challenge2011' / 'set-a'


def get_dataset(data_dir: str | Path | None = None) -> ECGDataset:
    """
    Get ECGDataset with default or custom path.

    Args:
        data_dir: Optional custom path to set-a directory

    Returns:
        ECGDataset instance
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    return ECGDataset(data_dir)
