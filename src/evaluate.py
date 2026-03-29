"""
evaluate.py
-----------
Evaluates the fault detection system against NASA CMAPSS ground-truth labels.

Metrics reported
----------------
- Precision, Recall, F1-Score
- Confusion matrix
- ROC-AUC curve
- Error-vs-RUL scatter plot (shows how reconstruction error correlates with
  engine health degradation — key insight for PHM applications)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Classification metrics ───────────────────────────────────────────────────

def print_classification_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    errors:      np.ndarray,
) -> dict:
    """
    Print a full fault-detection classification report.

    Parameters
    ----------
    true_labels : (N,) int — 1 = truly degraded, 0 = healthy
    pred_labels : (N,) bool/int — 1 = anomaly detected, 0 = normal
    errors      : (N,) float — raw reconstruction errors (for AUC)
    """
    prec  = precision_score(true_labels, pred_labels, zero_division=0)
    rec   = recall_score(   true_labels, pred_labels, zero_division=0)
    f1    = f1_score(       true_labels, pred_labels, zero_division=0)

    try:
        auc = roc_auc_score(true_labels, errors)
    except ValueError:
        auc = float("nan")

    print("\n" + "═" * 50)
    print("  FAULT DETECTION EVALUATION REPORT")
    print("═" * 50)
    print(f"  Precision  : {prec:.4f}  — of flagged engines, how many are truly degraded")
    print(f"  Recall     : {rec:.4f}  — of truly degraded engines, how many were caught")
    print(f"  F1-Score   : {f1:.4f}  — harmonic mean of Precision and Recall")
    print(f"  ROC-AUC    : {auc:.4f}  — area under the ROC curve (1.0 = perfect)")
    print("─" * 50)

    n_true_pos  = int(((true_labels == 1) & (pred_labels == 1)).sum())
    n_true_neg  = int(((true_labels == 0) & (pred_labels == 0)).sum())
    n_false_pos = int(((true_labels == 0) & (pred_labels == 1)).sum())
    n_false_neg = int(((true_labels == 1) & (pred_labels == 0)).sum())

    print(f"  True  Positives (correctly flagged faults)   : {n_true_pos}")
    print(f"  True  Negatives (correctly cleared healthy)  : {n_true_neg}")
    print(f"  False Positives (healthy flagged as fault)   : {n_false_pos}")
    print(f"  False Negatives (faults missed)              : {n_false_neg}")
    print("═" * 50 + "\n")

    return {
        "precision" : prec,
        "recall"    : rec,
        "f1"        : f1,
        "auc"       : auc,
        "tp"        : n_true_pos,
        "tn"        : n_true_neg,
        "fp"        : n_false_pos,
        "fn"        : n_false_neg,
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    save_path:   Path = None,
) -> None:
    cm   = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Healthy", "Degraded"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Fault Detection — Confusion Matrix")
    plt.tight_layout()
    out = save_path or RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {out}")


def plot_roc_curve(
    true_labels: np.ndarray,
    errors:      np.ndarray,
    save_path:   Path = None,
) -> None:
    fpr, tpr, _ = roc_curve(true_labels, errors)
    auc_score   = roc_auc_score(true_labels, errors)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#1565C0", linewidth=2,
            label=f"ROC curve (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1565C0")
    ax.set_xlabel("False Positive Rate (Healthy → Flagged)")
    ax.set_ylabel("True Positive Rate (Degraded → Caught)")
    ax.set_title("Fault Detection — ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = save_path or RESULTS_DIR / "roc_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] ROC curve saved → {out}")


def plot_error_vs_rul(
    errors:      np.ndarray,
    rul_gt:      np.ndarray,
    threshold:   float,
    save_path:   Path = None,
) -> None:
    """
    Scatter: reconstruction error vs. ground-truth RUL.
    Healthy = blue, degraded (detected) = red, missed = orange.
    Shows whether the model's error signal correlates with engine health.
    """
    FAULT_RUL_CUTOFF = 50
    true_degraded = rul_gt <= FAULT_RUL_CUTOFF
    detected      = errors > threshold

    colors = np.where(
        true_degraded & detected,    "#C62828",   # True positive — red
        np.where(
            true_degraded & ~detected, "#F57C00",  # False negative — orange
            np.where(
                ~true_degraded & detected, "#AD1457",  # False positive — pink
                "#1565C0"                               # True negative — blue
            )
        )
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(rul_gt, errors, c=colors, s=20, alpha=0.75, linewidths=0)
    ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Threshold = {threshold:.4f}")
    ax.axvline(FAULT_RUL_CUTOFF, color="gray", linestyle=":", linewidth=1,
               label=f"RUL = {FAULT_RUL_CUTOFF} (fault label boundary)")
    ax.set_xlabel("Ground-truth RUL (remaining cycles to failure)")
    ax.set_ylabel("Reconstruction Error (MSE)")
    ax.set_title("Reconstruction Error vs. Remaining Useful Life")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Custom legend markers
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#C62828', markersize=7, label='True Positive'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#F57C00', markersize=7, label='False Negative'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#AD1457', markersize=7, label='False Positive'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#1565C0', markersize=7, label='True Negative'),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out = save_path or RESULTS_DIR / "error_vs_rul.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Error-vs-RUL plot saved → {out}")


def plot_error_distribution(
    errors_healthy:  np.ndarray,
    errors_degraded: np.ndarray,
    threshold:       float,
    save_path:       Path = None,
) -> None:
    """
    Histogram overlay: healthy vs degraded reconstruction error distributions.
    The threshold should separate these two populations.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    bins = np.linspace(0, max(errors_healthy.max(), errors_degraded.max()) * 1.1, 60)

    ax.hist(errors_healthy,  bins=bins, alpha=0.6, color="#1565C0",
            label="Healthy windows",   density=True)
    ax.hist(errors_degraded, bins=bins, alpha=0.6, color="#C62828",
            label="Degraded windows", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Reconstruction Error Distribution: Healthy vs Degraded")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = save_path or RESULTS_DIR / "error_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[evaluate] Error distribution saved → {out}")


# ─── Full evaluation pipeline ─────────────────────────────────────────────────

def run_evaluation(results: dict, train_X_healthy=None, train_X_degraded=None) -> dict:
    """
    Run full evaluation given the detection results dict from detect.py.

    Parameters
    ----------
    results          : output of detect.run_detection_pipeline()
    train_X_healthy  : healthy training windows (optional, for distribution plot)
    train_X_degraded : degraded training windows (optional)
    """
    errors      = results["errors"]
    is_anomaly  = results["is_anomaly"].astype(int)
    true_labels = results["true_labels"]
    threshold   = results["threshold"]
    rul_gt      = results["rul_gt"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = print_classification_report(true_labels, is_anomaly, errors)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(true_labels, is_anomaly)

    try:
        plot_roc_curve(true_labels, errors)
    except ValueError as e:
        print(f"[evaluate] Skipping ROC curve: {e}")

    plot_error_vs_rul(errors, rul_gt, threshold)

    if train_X_healthy is not None and train_X_degraded is not None:
        from detect import score_windows
        from model  import load_model
        model  = load_model()
        h_err  = score_windows(model, train_X_healthy)
        d_err  = score_windows(model, train_X_degraded)
        plot_error_distribution(h_err, d_err, threshold)

    return metrics


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from detect import run_detection_pipeline
    from data_loader import load_train, download_dataset
    from preprocessor import prepare_training_data

    download_dataset()
    train_df = load_train("FD001")
    X_healthy, X_degraded, _, _ = prepare_training_data(train_df)

    results = run_detection_pipeline()
    run_evaluation(results, X_healthy, X_degraded)
