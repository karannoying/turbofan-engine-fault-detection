"""
data_loader.py
--------------
Downloads and parses the NASA CMAPSS turbofan engine dataset.

The CMAPSS dataset contains run-to-failure sensor readings from turbofan
engines under different operating conditions and fault modes.

Sub-datasets:
  FD001 - 1 operating condition, 1 fault mode  (simplest, best to start)
  FD002 - 6 operating conditions, 1 fault mode
  FD003 - 1 operating condition, 2 fault modes
  FD004 - 6 operating conditions, 2 fault modes
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path


# ─── Dataset columns ──────────────────────────────────────────────────────────

# 3 operational settings + 21 sensors
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
OP_COLS     = ["op1", "op2", "op3"]

COLUMN_NAMES = ["engine_id", "cycle"] + OP_COLS + SENSOR_COLS

# Sensors that are constant / near-zero variance across all engines in FD001
# — these add noise so we drop them before training
DROPPED_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

FEATURE_COLS = [c for c in SENSOR_COLS + OP_COLS if c not in DROPPED_SENSORS]


# ─── Download helpers ─────────────────────────────────────────────────────────

DATA_URL  = "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip"
DATA_DIR  = Path(__file__).parent.parent / "data"
ZIP_PATH  = DATA_DIR / "cmapss.zip"


def download_dataset(force: bool = False) -> None:
    """
    Download the CMAPSS zip from NASA's open data portal.
    Falls back gracefully if the URL is unreachable and local files exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    if not force and (DATA_DIR / "train_FD001.txt").exists():
        print("[data_loader] Dataset already present — skipping download.")
        return

    print("[data_loader] Downloading NASA CMAPSS dataset …")
    try:
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)
        print("[data_loader] Download complete. Extracting …")
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(DATA_DIR)
        ZIP_PATH.unlink(missing_ok=True)
        print("[data_loader] Extraction done.")
    except Exception as e:
        print(f"[data_loader] WARNING: Could not download dataset: {e}")
        print("[data_loader] Place the CMAPSS .txt files manually in ./data/")


def _read_txt(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited CMAPSS file into a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df = df.dropna(axis=1, how="all")   # drop trailing empty columns
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"]     = df["cycle"].astype(int)
    return df


def load_train(subset: str = "FD001") -> pd.DataFrame:
    """
    Load training data for a given CMAPSS subset.

    Each row is one engine cycle.  The training set runs each engine
    until failure (last cycle = end-of-life).

    Returns
    -------
    pd.DataFrame with columns:
        engine_id, cycle, op1-3, s1-21, rul
        where `rul` = Remaining Useful Life (cycles to failure)
    """
    path = DATA_DIR / f"train_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run download_dataset() first, or place "
            f"the CMAPSS files in {DATA_DIR}."
        )

    df = _read_txt(path)

    # Compute RUL: for each engine, max_cycle - current_cycle
    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)

    return df


def load_test(subset: str = "FD001") -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load test data and corresponding RUL ground-truth labels.

    Returns
    -------
    df     : pd.DataFrame  — sensor readings (no RUL column)
    rul_gt : np.ndarray    — true RUL at last observed cycle, one per engine
    """
    test_path = DATA_DIR / f"test_{subset}.txt"
    rul_path  = DATA_DIR / f"RUL_{subset}.txt"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}.")

    df     = _read_txt(test_path)
    rul_gt = pd.read_csv(rul_path, header=None).values.flatten().astype(int)

    return df, rul_gt


def describe_dataset(df: pd.DataFrame) -> None:
    """Print a quick summary of a loaded dataframe."""
    n_engines = df["engine_id"].nunique()
    n_cycles  = df.groupby("engine_id")["cycle"].max()
    print(f"  Engines   : {n_engines}")
    print(f"  Cycles    : min={n_cycles.min()}, mean={n_cycles.mean():.0f}, max={n_cycles.max()}")
    print(f"  Rows      : {len(df):,}")
    print(f"  Features  : {FEATURE_COLS}")


# ─── Quick smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_dataset()
    train_df = load_train("FD001")
    print("\nTraining set:")
    describe_dataset(train_df)

    test_df, rul_gt = load_test("FD001")
    print("\nTest set:")
    describe_dataset(test_df)
    print(f"\nRUL ground truth (first 10): {rul_gt[:10]}")
