"""
preprocessor.py
---------------
Transforms raw CMAPSS data into normalised sliding-window sequences
ready for the LSTM Autoencoder.

Pipeline
--------
1. Min-max normalise each feature column (fit on healthy training data only)
2. Optionally smooth sensors with a rolling mean
3. Build overlapping sliding windows of length `window_size`
4. Split training windows into "healthy" (RUL > healthy_threshold) and
   "degraded" subsets — we train only on healthy windows so the autoencoder
   learns a clean baseline.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from data_loader import FEATURE_COLS, DROPPED_SENSORS


# ─── Defaults ─────────────────────────────────────────────────────────────────

WINDOW_SIZE        = 30     # cycles per sequence
STEP_SIZE          = 1      # stride between consecutive windows
HEALTHY_THRESHOLD  = 125    # RUL (cycles) above which an engine is "healthy"
SMOOTH_WINDOW      = 5      # rolling-mean smoothing window (None = off)

SCALER_PATH = Path(__file__).parent.parent / "models" / "scaler.pkl"


# ─── Normalisation ────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, features: list[str] = FEATURE_COLS) -> MinMaxScaler:
    """Fit a MinMaxScaler on healthy training cycles only."""
    healthy = df[df["rul"] > HEALTHY_THRESHOLD]
    scaler  = MinMaxScaler()
    scaler.fit(healthy[features])
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: MinMaxScaler,
    features: list[str] = FEATURE_COLS,
) -> pd.DataFrame:
    """Return a copy of df with the given features scaled to [0, 1]."""
    df = df.copy()
    df[features] = scaler.transform(df[features])
    return df


def save_scaler(scaler: MinMaxScaler, path: Path = SCALER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[preprocessor] Scaler saved → {path}")


def load_scaler(path: Path = SCALER_PATH) -> MinMaxScaler:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Smoothing ────────────────────────────────────────────────────────────────

def smooth_sensors(
    df: pd.DataFrame,
    features: list[str] = FEATURE_COLS,
    window: int = SMOOTH_WINDOW,
) -> pd.DataFrame:
    """
    Apply per-engine rolling mean to reduce high-frequency sensor noise.
    Fills NaNs at the start of each engine's history with forward fill.
    """
    if window is None or window <= 1:
        return df

    df = df.copy()

    def _smooth_group(g: pd.DataFrame) -> pd.DataFrame:
        g[features] = (
            g[features]
            .rolling(window=window, min_periods=1)
            .mean()
        )
        return g

    df = df.groupby("engine_id", group_keys=False).apply(_smooth_group)
    return df


# ─── Sliding windows ──────────────────────────────────────────────────────────

def make_windows(
    df: pd.DataFrame,
    features: list[str] = FEATURE_COLS,
    window_size: int     = WINDOW_SIZE,
    step_size: int       = STEP_SIZE,
    include_rul: bool    = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build sliding-window sequences from per-engine cycle data.

    Parameters
    ----------
    df          : DataFrame with columns engine_id, cycle, features, [rul]
    features    : sensor/op columns to include in each window
    window_size : number of cycles per window
    step_size   : stride between windows
    include_rul : if True, also return the RUL at the last cycle of each window

    Returns
    -------
    X    : np.ndarray  shape (N, window_size, n_features)
    rul  : np.ndarray  shape (N,)  — only if include_rul=True, else None
    """
    windows, ruls = [], []

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")
        values = group[features].values          # (T, F)

        rul_vals = group["rul"].values if "rul" in group.columns else None

        for start in range(0, len(values) - window_size + 1, step_size):
            end = start + window_size
            windows.append(values[start:end])
            if rul_vals is not None:
                ruls.append(rul_vals[end - 1])   # RUL at last cycle of window

    X   = np.array(windows, dtype=np.float32)         # (N, W, F)
    rul = np.array(ruls, dtype=np.float32) if ruls else None

    return X, rul


def make_last_windows(
    df: pd.DataFrame,
    features: list[str] = FEATURE_COLS,
    window_size: int     = WINDOW_SIZE,
) -> np.ndarray:
    """
    For test engines: extract only the LAST window_size cycles.
    Returns shape (n_engines, window_size, n_features).
    """
    windows = []
    for engine_id, group in df.groupby("engine_id"):
        group  = group.sort_values("cycle")
        values = group[features].values
        if len(values) >= window_size:
            windows.append(values[-window_size:])
        else:
            # Pad with the first row if the engine has fewer cycles than the window
            pad_len = window_size - len(values)
            padded  = np.vstack([
                np.tile(values[0], (pad_len, 1)),
                values
            ])
            windows.append(padded)

    return np.array(windows, dtype=np.float32)


# ─── Train / healthy split ────────────────────────────────────────────────────

def split_healthy_degraded(
    X: np.ndarray,
    rul: np.ndarray,
    healthy_threshold: int = HEALTHY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split windows into healthy (RUL > threshold) and degraded (RUL <= threshold).

    The autoencoder is trained ONLY on healthy windows so it learns to
    reconstruct normal engine behaviour. Degraded windows are used to
    verify that reconstruction error rises during fault progression.
    """
    mask_healthy  = rul > healthy_threshold
    mask_degraded = ~mask_healthy

    X_healthy  = X[mask_healthy]
    X_degraded = X[mask_degraded]

    print(
        f"[preprocessor] Healthy windows : {len(X_healthy):>6,}  "
        f"(RUL > {healthy_threshold})"
    )
    print(
        f"[preprocessor] Degraded windows: {len(X_degraded):>6,}  "
        f"(RUL ≤ {healthy_threshold})"
    )

    return X_healthy, X_degraded


# ─── Full pipeline helper ─────────────────────────────────────────────────────

def prepare_training_data(
    train_df: pd.DataFrame,
    features: list[str] = FEATURE_COLS,
    window_size: int     = WINDOW_SIZE,
    smooth: bool         = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    One-shot: normalise → smooth → window → split.

    Returns
    -------
    X_healthy   : (N_h, W, F) — used for autoencoder training
    X_degraded  : (N_d, W, F) — held out for threshold calibration checks
    rul_all     : (N,)         — RUL labels for all windows
    scaler      : fitted MinMaxScaler
    """
    scaler   = fit_scaler(train_df, features)
    scaled   = apply_scaler(train_df, scaler, features)

    if smooth:
        scaled = smooth_sensors(scaled, features)

    X_all, rul_all = make_windows(scaled, features, window_size, include_rul=True)
    X_healthy, X_degraded = split_healthy_degraded(X_all, rul_all)

    return X_healthy, X_degraded, rul_all, scaler


def prepare_test_data(
    test_df: pd.DataFrame,
    scaler: MinMaxScaler,
    features: list[str] = FEATURE_COLS,
    window_size: int     = WINDOW_SIZE,
    smooth: bool         = True,
) -> np.ndarray:
    """
    Normalise and extract the last-window for every test engine.
    Returns shape (n_engines, window_size, n_features).
    """
    scaled = apply_scaler(test_df, scaler, features)
    if smooth:
        scaled = smooth_sensors(scaled, features)

    return make_last_windows(scaled, features, window_size)


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_loader import load_train, load_test, download_dataset

    download_dataset()
    train_df = load_train("FD001")
    test_df, rul_gt = load_test("FD001")

    X_healthy, X_degraded, rul_all, scaler = prepare_training_data(train_df)
    print(f"\nX_healthy  shape: {X_healthy.shape}")
    print(f"X_degraded shape: {X_degraded.shape}")

    X_test = prepare_test_data(test_df, scaler)
    print(f"X_test     shape: {X_test.shape}")

    save_scaler(scaler)
