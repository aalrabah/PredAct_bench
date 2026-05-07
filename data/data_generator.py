import os
import zipfile
import urllib.request
from pathlib import Path

OULAD_URL = "https://analyse.kmi.open.ac.uk/open-dataset/download"

# Local paths
DATA_DIR = Path("data/oulad")
ZIP_PATH = DATA_DIR / "oulad.zip"


def download_oulad():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"[skip] {ZIP_PATH} already exists.")
    else:
        print(f"[download] {OULAD_URL} -> {ZIP_PATH}")
        urllib.request.urlretrieve(OULAD_URL, ZIP_PATH)
        print(f"[done] downloaded {ZIP_PATH.stat().st_size / 1e6:.1f} MB")

    print(f"[extract] {ZIP_PATH} -> {DATA_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    csvs = sorted(DATA_DIR.glob("*.csv"))
    print(f"[done] extracted {len(csvs)} CSV files:")
    for c in csvs:
        print(f"  - {c.name}")


if __name__ == "__main__":
    download_oulad()