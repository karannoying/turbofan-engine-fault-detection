"""
run.py
------
Master entry point — runs the full project pipeline in one command:

    python run.py

Steps
-----
1.  Download NASA CMAPSS dataset (if not already present)
2.  Preprocess: normalise, smooth, sliding windows
3.  Train LSTM Autoencoder on healthy engine windows
4.  Calibrate anomaly threshold on healthy training errors
5.  Detect faults on the test set
6.  Evaluate and generate all plots into ./results/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import time

from data_loader  import download_dataset, load_train, load_test
from preprocessor import prepare_training_data, prepare_test_data, load_scaler
from train        import train
from detect       import (
    run_detection_pipeline,
    calibrate_threshold,
    save_threshold,
    load_threshold,
    detect_faults,
    plot_engine_errors,
    score_windows,
)
from evaluate     import run_evaluation, plot_error_distribution
from model        import load_model


def banner(text: str) -> None:
    line = "─" * 60
    print(f"\n{line}")
    print(f"  {text}")
    print(f"{line}")


def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    subset = args.subset

    # ── Step 1 : Data ─────────────────────────────────────────────────────────
    banner("Step 1/5 — Downloading & loading dataset")
    download_dataset()
    train_df        = load_train(subset)
    test_df, rul_gt = load_test(subset)
    print(f"  Train rows : {len(train_df):,}")
    print(f"  Test rows  : {len(test_df):,}")

    # ── Step 2 : Preprocess ───────────────────────────────────────────────────
    banner("Step 2/5 — Preprocessing")
    X_healthy, X_degraded, rul_all, scaler = prepare_training_data(
        train_df,
        window_size=args.window_size,
    )
    X_test = prepare_test_data(test_df, scaler, window_size=args.window_size)
    print(f"  X_healthy shape : {X_healthy.shape}")
    print(f"  X_test    shape : {X_test.shape}")

    # ── Step 3 : Train ────────────────────────────────────────────────────────
    if args.skip_training:
        banner("Step 3/5 — Loading existing model (--skip-training)")
        model = load_model()
    else:
        banner("Step 3/5 — Training LSTM Autoencoder")
        model = train(
            subset      = subset,
            epochs      = args.epochs,
            batch_size  = args.batch_size,
            lr          = args.lr,
            window_size = args.window_size,
        )

    # ── Step 4 : Threshold ────────────────────────────────────────────────────
    banner("Step 4/5 — Calibrating anomaly threshold")

    device = "cpu"
    try:
        import torch
        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
    except ImportError:
        pass

    threshold = calibrate_threshold(
        model, X_healthy,
        percentile=args.threshold_percentile,
        device=device,
    )
    save_threshold(threshold)

    # ── Step 5 : Detect + Evaluate ────────────────────────────────────────────
    banner("Step 5/5 — Detecting faults & evaluating")

    errors, is_anomaly = detect_faults(model, X_test, threshold, device=device)

    FAULT_RUL_CUTOFF = 50
    import numpy as np
    true_labels = (rul_gt <= FAULT_RUL_CUTOFF).astype(int)

    results = {
        "errors"     : errors,
        "is_anomaly" : is_anomaly,
        "true_labels": true_labels,
        "threshold"  : threshold,
        "rul_gt"     : rul_gt,
    }

    # Generate engine error curves on scaled training data
    from preprocessor import apply_scaler, smooth_sensors, FEATURE_COLS
    scaled_train = apply_scaler(train_df, scaler)
    scaled_train = smooth_sensors(scaled_train)
    plot_engine_errors(model, scaled_train, threshold, device=device)

    # Error distribution plot (healthy vs degraded training windows)
    h_err = score_windows(model, X_healthy, device=device)
    d_err = score_windows(model, X_degraded, device=device)
    plot_error_distribution(h_err, d_err, threshold)

    # Full metrics + ROC + confusion matrix
    metrics = run_evaluation(results)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner("Done")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {metrics['auc']:.4f}")
    print(f"  Elapsed   : {elapsed:.1f}s")
    print(f"\n  All plots saved to → ./results/")
    print(f"  Run the dashboard  → streamlit run app.py\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Turbofan Engine Fault Detection — full pipeline"
    )
    p.add_argument("--subset",                default="FD001",
                   help="CMAPSS subset: FD001, FD002, FD003, FD004")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch-size",  type=int, default=64, dest="batch_size")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--window-size", type=int,   default=30, dest="window_size")
    p.add_argument("--threshold-percentile", type=float, default=95.0,
                   dest="threshold_percentile",
                   help="Percentile of healthy errors used as anomaly threshold")
    p.add_argument("--skip-training", action="store_true", dest="skip_training",
                   help="Skip training and load an existing saved model")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
