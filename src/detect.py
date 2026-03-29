"""
detect.py
---------
Anomaly detection via reconstruction error from the trained LSTM Autoencoder.

Core idea
---------
1. Feed sensor windows through the autoencoder
2. Compute MSE between input and reconstruction for each window
3. A threshold (95th percentile of healthy training errors) separates
   normal from anomalous behaviour
4. Any window with error > threshold is flagged as a fault

This module handles:
  - Scoring individual windows or full engine sequences
  - Threshold calibration from healthy training data
  - Per-engine error-vs-cycle plots showing fault progression
  - Saving/loading the threshold value
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_loader  import load_train, load_test, download_dataset, FEATURE_COLS
from preprocessor import (
    prepare_training_data, prepare_test_data,
    load_scaler, make_windows, apply_scaler, smooth_sensors,
    WINDOW_SIZE, HEALTHY_THRESHOLD
)
from model import LSTMAutoencoder, load_model


# ─── Paths ────────────────────────────────────────────────────────────────────

MODELS_DIR    = Path(__file__).parent.parent / "models"
RESULTS_DIR   = Path(__file__).parent.parent / "results"
THRESHOLD_PATH = MODELS_DIR / "threshold.pkl"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Scoring ──────────────────────────────────────────────────────────────────

def score_windows(
    model:  LSTMAutoencoder,
    X:      np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """
    Compute MSE reconstruction error for a numpy array of windows.

    Parameters
    ----------
    model  : trained LSTMAutoencoder
    X      : (N, window_size, n_features)
    device : "cpu" | "cuda" | "mps"
    batch_size : mini-batch size for inference

    Returns
    -------
    errors : (N,) float32 — one reconstruction error per window
    """
    model.eval()
    dev    = torch.device(device)
    model  = model.to(dev)
    errors = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i : i + batch_size]).to(dev)
            err   = model.reconstruction_error(batch)
            errors.append(err.cpu().numpy())

    return np.concatenate(errors).astype(np.float32)


# ─── Threshold ────────────────────────────────────────────────────────────────

def calibrate_threshold(
    model:      LSTMAutoencoder,
    X_healthy:  np.ndarray,
    percentile: float = 95.0,
    device:     str   = "cpu",
) -> float:
    """
    Set the anomaly threshold at the Nth percentile of healthy reconstruction errors.
    Anything above this is classified as anomalous.
    """
    errors    = score_windows(model, X_healthy, device=device)
    threshold = float(np.percentile(errors, percentile))
    print(
        f"[detect] Threshold @ {percentile:.0f}th pct = {threshold:.6f}  "
        f"(healthy error mean={errors.mean():.6f}, std={errors.std():.6f})"
    )
    return threshold


def save_threshold(threshold: float, path: Path = THRESHOLD_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(threshold, f)
    print(f"[detect] Threshold saved → {path}")


def load_threshold(path: Path = THRESHOLD_PATH) -> float:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Per-engine fault progression ─────────────────────────────────────────────

def score_engine_over_time(
    model:       LSTMAutoencoder,
    train_df,                          # raw (scaled) DataFrame for one engine
    features:    list = FEATURE_COLS,
    window_size: int  = WINDOW_SIZE,
    device:      str  = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Score every sliding window for a single engine to show how reconstruction
    error evolves as the engine approaches failure.

    Returns
    -------
    cycles : (N,)  — cycle index at the end of each window
    errors : (N,)  — reconstruction MSE at that cycle
    """
    from preprocessor import make_windows

    X, _ = make_windows(
        train_df, features=features, window_size=window_size, include_rul=False
    )

    if len(X) == 0:
        return np.array([]), np.array([])

    errors = score_windows(model, X, device=device)

    # Align cycle numbers (end of each window)
    cycles = np.arange(window_size, window_size + len(errors))
    return cycles, errors


def plot_engine_errors(
    model:     LSTMAutoencoder,
    train_df,
    threshold: float,
    engine_ids: list   = None,
    n_engines:  int    = 6,
    features:   list   = FEATURE_COLS,
    window_size: int   = WINDOW_SIZE,
    device:     str    = "cpu",
    save_path:  Path   = None,
) -> None:
    """
    Grid plot: reconstruction error vs. cycle for a sample of engines.
    Red shading shows the degraded zone (RUL <= HEALTHY_THRESHOLD).
    Red dashed line = anomaly threshold.
    """
    from preprocessor import apply_scaler, smooth_sensors

    all_engines = train_df["engine_id"].unique()
    if engine_ids is None:
        rng        = np.random.default_rng(42)
        engine_ids = rng.choice(all_engines, size=min(n_engines, len(all_engines)), replace=False)

    n_cols = 3
    n_rows = int(np.ceil(len(engine_ids) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for ax, eid in zip(axes, engine_ids):
        eng_df = train_df[train_df["engine_id"] == eid].copy()
        cycles, errors = score_engine_over_time(
            model, eng_df, features=features, window_size=window_size, device=device
        )
        if len(cycles) == 0:
            ax.set_visible(False)
            continue

        ax.plot(cycles, errors, color="#1565C0", linewidth=1.2, label="Recon error")
        ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2, label="Threshold")

        # Mark degraded zone
        max_cycle = eng_df["cycle"].max()
        deg_start = max(cycles[0], max_cycle - HEALTHY_THRESHOLD)
        ax.axvspan(deg_start, max_cycle + 1, color="crimson", alpha=0.08, label="Degraded zone")

        ax.set_title(f"Engine {eid}", fontsize=10)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("MSE")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for ax in axes[len(engine_ids):]:
        ax.set_visible(False)

    # Shared legend
    handles = [
        mpatches.Patch(color="#1565C0",   label="Reconstruction error"),
        mpatches.Patch(color="crimson",   label="Threshold", alpha=0.8),
        mpatches.Patch(color="crimson",   label="Degraded zone", alpha=0.15),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Reconstruction Error vs. Engine Cycle", fontsize=13, y=1.01)
    plt.tight_layout()

    out = save_path or RESULTS_DIR / "engine_error_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[detect] Engine error curves saved → {out}")


# ─── Test-set fault detection ─────────────────────────────────────────────────

def detect_faults(
    model:     LSTMAutoencoder,
    X_test:    np.ndarray,
    threshold: float,
    device:    str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run anomaly detection on test windows.

    Returns
    -------
    errors     : (N,) reconstruction errors
    is_anomaly : (N,) boolean array — True = fault detected
    """
    errors     = score_windows(model, X_test, device=device)
    is_anomaly = errors > threshold
    n_flagged  = is_anomaly.sum()
    print(
        f"[detect] {n_flagged}/{len(errors)} engines flagged as anomalous "
        f"({100*n_flagged/len(errors):.1f}%)"
    )
    return errors, is_anomaly


# ─── Full detection pipeline ──────────────────────────────────────────────────

def run_detection_pipeline(subset: str = "FD001") -> dict:
    """
    End-to-end:
      load data → preprocess → calibrate threshold → detect on test set

    Returns a dict of results for downstream evaluation.
    """
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # ── Load ──────────────────────────────────────────────────────────────────
    download_dataset()
    train_df       = load_train(subset)
    test_df, rul_gt = load_test(subset)
    scaler         = load_scaler()
    model          = load_model(device=device)

    # ── Preprocess ────────────────────────────────────────────────────────────
    X_healthy, _, _, _ = prepare_training_data(train_df)

    scaled_train = apply_scaler(train_df, scaler)
    scaled_train = smooth_sensors(scaled_train)

    X_test = prepare_test_data(test_df, scaler)

    # ── Threshold ─────────────────────────────────────────────────────────────
    threshold = calibrate_threshold(model, X_healthy, device=device)
    save_threshold(threshold)

    # ── Detect ────────────────────────────────────────────────────────────────
    errors, is_anomaly = detect_faults(model, X_test, threshold, device=device)

    # ── Visualise ─────────────────────────────────────────────────────────────
    plot_engine_errors(model, scaled_train, threshold, device=device)

    # ── Ground truth labels ───────────────────────────────────────────────────
    # An engine is "truly degraded" if its remaining useful life is short
    FAULT_RUL_CUTOFF = 50
    true_labels = (rul_gt <= FAULT_RUL_CUTOFF).astype(int)

    return {
        "errors"      : errors,
        "is_anomaly"  : is_anomaly,
        "true_labels" : true_labels,
        "threshold"   : threshold,
        "rul_gt"      : rul_gt,
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_detection_pipeline()
    print("\nDetection complete.  Errors (first 10):", results["errors"][:10])
