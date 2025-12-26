"""
Download PhysioNet Challenge 2011 Dataset
ECG Quality Classification (set-a training data)
"""

from pathlib import Path
import tarfile
import requests
import pandas as pd
import wfdb
from tqdm import tqdm


def download_file(url: str, dest_path: Path) -> None:
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_challenge_2011(data_dir: Path) -> None:
    """Download PhysioNet Challenge 2011 set-a dataset."""

    url = "https://physionet.org/static/published-projects/challenge-2011/improving-the-quality-of-ecgs-collected-using-mobile-phones-the-physionetcomputing-in-cardiology-challenge-2011-1.0.0.zip"

    # Alternative: direct tar.gz
    tar_url = "https://www.physionet.org/files/challenge-2011/1.0.0/set-a.tar.gz"

    target_dir = data_dir / "challenge2011"
    target_dir.mkdir(parents=True, exist_ok=True)

    tar_path = target_dir / "set-a.tar.gz"

    print(f"Downloading PhysioNet Challenge 2011 set-a (~103MB)...")
    download_file(tar_url, tar_path)

    print("Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(target_dir)

    # Clean up tar file
    tar_path.unlink()
    print(f"Dataset extracted to {target_dir / 'set-a'}")


def create_labels_csv(data_dir: Path) -> pd.DataFrame:
    """Parse RECORDS files and create reference_labels.csv."""

    set_a_dir = data_dir / "challenge2011" / "set-a"

    records = []

    # Read acceptable records
    acceptable_file = set_a_dir / "RECORDS-acceptable"
    if acceptable_file.exists():
        with open(acceptable_file, 'r') as f:
            for line in f:
                record_id = line.strip()
                if record_id:
                    records.append({'record_id': record_id, 'quality': 'Acceptable'})

    # Read unacceptable records
    unacceptable_file = set_a_dir / "RECORDS-unacceptable"
    if unacceptable_file.exists():
        with open(unacceptable_file, 'r') as f:
            for line in f:
                record_id = line.strip()
                if record_id:
                    records.append({'record_id': record_id, 'quality': 'Unacceptable'})

    df = pd.DataFrame(records)

    # Save to CSV
    csv_path = set_a_dir / "reference_labels.csv"
    df.to_csv(csv_path, index=False)
    print(f"Labels saved to {csv_path}")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print dataset summary statistics."""

    total = len(df)
    acceptable = len(df[df['quality'] == 'Acceptable'])
    unacceptable = len(df[df['quality'] == 'Unacceptable'])

    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Total records:  {total}")
    print(f"Acceptable:     {acceptable} ({100*acceptable/total:.1f}%)")
    print(f"Unacceptable:   {unacceptable} ({100*unacceptable/total:.1f}%)")
    print("="*50)


def test_load_record(data_dir: Path, record_id: str) -> None:
    """Test loading a sample ECG record."""

    set_a_dir = data_dir / "challenge2011" / "set-a"
    record_path = str(set_a_dir / record_id)

    print(f"\nTesting ECG load: {record_id}")

    record = wfdb.rdrecord(record_path)

    print(f"  Sampling frequency: {record.fs} Hz")
    print(f"  Duration: {record.sig_len / record.fs:.1f} seconds")
    print(f"  Leads: {record.n_sig} ({', '.join(record.sig_name)})")
    print(f"  Shape: {record.p_signal.shape}")
    print("  Load successful!")


def main():
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / "data"

    # Step 1: Download dataset
    download_challenge_2011(data_dir)

    # Step 2: Create labels CSV
    df = create_labels_csv(data_dir)

    # Step 3: Print summary
    print_summary(df)

    # Step 4: Test loading one record
    if len(df) > 0:
        test_record = df.iloc[0]['record_id']
        test_load_record(data_dir, test_record)

    print("\nSetup complete! Ready to start analysis.")


if __name__ == "__main__":
    main()
