"""
app.py
------
Streamlit dashboard for the Turbofan Engine Fault Detection system.

Run with:
    streamlit run app.py

Features
--------
- System overview with live metrics (Precision / Recall / F1 / AUC)
- Engine selector: plot error-over-time for any individual engine
- Threshold slider: adjust the anomaly threshold and see how metrics change in real time
- Test-set results table: per-engine anomaly scores and fault status
- Error distribution chart
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "Turbofan Fault Detection",
    page_icon   = "✈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)


# ─── Helpers / caching ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def get_model():
    from model import load_model
    return load_model()


@st.cache_data(show_spinner="Loading dataset …")
def get_data(subset="FD001"):
    from data_loader  import load_train, load_test, download_dataset
    download_dataset()
    train_df        = load_train(subset)
    test_df, rul_gt = load_test(subset)
    return train_df, test_df, rul_gt


@st.cache_data(show_spinner="Preprocessing …")
def get_preprocessed(subset="FD001"):
    from preprocessor import (
        prepare_training_data, prepare_test_data,
        apply_scaler, smooth_sensors, save_scaler, load_scaler, FEATURE_COLS
    )
    from model import load_model
    from pathlib import Path

    train_df, test_df, rul_gt = get_data(subset)

    scaler_path = Path(__file__).parent / "models" / "scaler.pkl"
    if scaler_path.exists():
        scaler = load_scaler(scaler_path)
        X_healthy, X_degraded, rul_all, _ = prepare_training_data(train_df)
    else:
        X_healthy, X_degraded, rul_all, scaler = prepare_training_data(train_df)
        save_scaler(scaler, scaler_path)

    X_test        = prepare_test_data(test_df, scaler)
    scaled_train  = apply_scaler(train_df, scaler)
    scaled_train  = smooth_sensors(scaled_train)

    return X_healthy, X_degraded, X_test, rul_gt, scaler, scaled_train


@st.cache_data(show_spinner="Scoring windows …")
def get_test_errors():
    from detect import score_windows
    model = get_model()
    _, _, X_test, rul_gt, _, _ = get_preprocessed()
    errors = score_windows(model, X_test)
    return errors, rul_gt


@st.cache_data(show_spinner="Scoring training windows …")
def get_train_errors():
    from detect import score_windows
    model = get_model()
    X_healthy, X_degraded, _, _, _, _ = get_preprocessed()
    h_err = score_windows(model, X_healthy)
    d_err = score_windows(model, X_degraded)
    return h_err, d_err


def compute_metrics(true_labels, is_anomaly, errors):
    prec = precision_score(true_labels, is_anomaly, zero_division=0)
    rec  = recall_score(   true_labels, is_anomaly, zero_division=0)
    f1   = f1_score(       true_labels, is_anomaly, zero_division=0)
    try:
        auc = roc_auc_score(true_labels, errors)
    except Exception:
        auc = float("nan")
    return prec, rec, f1, auc


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("✈ Turbofan Fault Detection")
    st.caption("NASA CMAPSS · LSTM Autoencoder · Anomaly Detection")
    st.divider()

    subset = st.selectbox(
        "Dataset subset",
        ["FD001", "FD002", "FD003", "FD004"],
        index=0,
        help="FD001 = 1 operating condition, easiest to train"
    )

    st.subheader("Anomaly Threshold")

    errors, rul_gt = get_test_errors()
    h_err, d_err   = get_train_errors()

    default_threshold = float(np.percentile(h_err, 95))

    threshold = st.slider(
        "Detection threshold",
        min_value  = float(h_err.min()),
        max_value  = float(max(h_err.max(), d_err.max()) * 1.05),
        value      = default_threshold,
        step       = (float(h_err.max()) - float(h_err.min())) / 200,
        format     = "%.5f",
        help       = "Windows with reconstruction error above this are flagged as anomalous"
    )

    FAULT_RUL_CUTOFF = 50
    true_labels  = (rul_gt <= FAULT_RUL_CUTOFF).astype(int)
    is_anomaly   = (errors > threshold).astype(int)
    prec, rec, f1, auc = compute_metrics(true_labels, is_anomaly, errors)

    st.divider()
    st.caption("Adjust the slider to see how threshold choice affects all metrics live.")


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Engine Inspector",
    "📈 Error Analysis",
    "📋 Test Results",
])


# ════════════════════════════════════════════════════════════════════════════
# Tab 1: Overview
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("System Performance Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision",  f"{prec:.3f}", help="Of flagged engines, fraction that are truly degraded")
    col2.metric("Recall",     f"{rec:.3f}",  help="Of truly degraded engines, fraction caught")
    col3.metric("F1-Score",   f"{f1:.3f}",   help="Harmonic mean of Precision and Recall")
    col4.metric("ROC-AUC",    f"{auc:.3f}",  help="Area under the ROC curve; 1.0 = perfect")

    st.divider()

    n_flagged  = is_anomaly.sum()
    n_engines  = len(errors)
    n_degraded = true_labels.sum()

    col5, col6, col7 = st.columns(3)
    col5.metric("Engines tested",          n_engines)
    col6.metric("Truly degraded engines",  n_degraded,
                help=f"Ground truth: RUL ≤ {FAULT_RUL_CUTOFF} cycles")
    col7.metric("Engines flagged as fault", n_flagged,
                delta=f"{int(n_flagged - n_degraded):+d} vs ground truth",
                delta_color="inverse")

    st.divider()
    st.subheader("How the system works")
    st.markdown("""
    1. **Data**: NASA CMAPSS — 21-sensor run-to-failure recordings from simulated turbofan engines.
    2. **Preprocessing**: Sensors normalised + smoothed → 30-cycle sliding windows.
    3. **Model**: LSTM Autoencoder trained *only* on healthy early-life windows. It learns to
       reconstruct normal engine patterns with low error.
    4. **Detection**: Engines in degraded states deviate from normal — reconstruction error **spikes**.
       Any window above the threshold is flagged as anomalous.
    5. **Evaluation**: Compared against NASA-provided Remaining Useful Life labels.
    """)


# ════════════════════════════════════════════════════════════════════════════
# Tab 2: Engine Inspector
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("Per-Engine Fault Progression")
    st.caption(
        "Reconstruction error rises as an engine degrades — the threshold crossing "
        "is the early warning signal."
    )

    train_df, _, _ = get_data(subset)
    _, _, _, _, scaler, scaled_train = get_preprocessed(subset)
    all_engine_ids = sorted(scaled_train["engine_id"].unique().tolist())

    selected_engine = st.selectbox(
        "Select engine", all_engine_ids, index=0,
        help="Each engine runs for a different number of cycles before failure"
    )

    from detect import score_engine_over_time
    from preprocessor import FEATURE_COLS, WINDOW_SIZE
    model = get_model()

    eng_df = scaled_train[scaled_train["engine_id"] == selected_engine].copy()
    cycles, eng_errors = score_engine_over_time(
        model, eng_df,
        features    = FEATURE_COLS,
        window_size = WINDOW_SIZE,
    )

    if len(cycles) == 0:
        st.warning("Not enough cycles for this engine.")
    else:
        max_cycle = eng_df["cycle"].max()
        rul_vals  = train_df[train_df["engine_id"] == selected_engine]["rul"].values
        min_rul   = rul_vals.min()

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)

        # ── Error over time ────────────────────────────────────────────────
        ax = axes[0]
        ax.plot(cycles, eng_errors, color="#1565C0", linewidth=1.5, label="Recon error")
        ax.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Threshold = {threshold:.4f}")
        from preprocessor import HEALTHY_THRESHOLD
        deg_start = max(cycles[0], max_cycle - HEALTHY_THRESHOLD)
        ax.axvspan(deg_start, max_cycle + 1, color="crimson", alpha=0.08,
                   label="Degraded zone")
        ax.set_ylabel("Reconstruction MSE")
        ax.set_title(f"Engine {selected_engine} — Reconstruction Error Over Time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Raw sensor (s2 — fan inlet temp, typically most informative) ──
        ax2 = axes[1]
        sensor_to_plot = "s2"
        if sensor_to_plot in eng_df.columns:
            ax2.plot(eng_df["cycle"], eng_df[sensor_to_plot],
                     color="#2E7D32", linewidth=1.2, label=f"Sensor {sensor_to_plot} (normalised)")
            ax2.axvspan(deg_start, max_cycle + 1, color="crimson", alpha=0.08)
            ax2.set_xlabel("Cycle")
            ax2.set_ylabel(f"Sensor {sensor_to_plot} (norm.)")
            ax2.set_title(f"Engine {selected_engine} — Raw Sensor Signal")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Summary stats
        first_alarm = cycles[eng_errors > threshold]
        if len(first_alarm) > 0:
            alarm_cycle   = int(first_alarm[0])
            cycles_before = max_cycle - alarm_cycle
            st.success(
                f"✅ First alarm at cycle **{alarm_cycle}** — "
                f"**{cycles_before} cycles before failure** (early warning!)"
            )
        else:
            st.info("No alarm triggered for this engine with the current threshold.")


# ════════════════════════════════════════════════════════════════════════════
# Tab 3: Error Analysis
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("Reconstruction Error Analysis")

    # ── Distribution ──────────────────────────────────────────────────────────
    st.subheader("Error Distribution: Healthy vs Degraded")
    st.caption("Good separation between the two distributions means the model has learnt the difference.")

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, max(h_err.max(), d_err.max()) * 1.1, 60)
    ax.hist(h_err, bins=bins, alpha=0.6, color="#1565C0", label="Healthy windows", density=True)
    ax.hist(d_err, bins=bins, alpha=0.6, color="#C62828", label="Degraded windows", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Reconstruction Error (MSE)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Error vs RUL ──────────────────────────────────────────────────────────
    st.subheader("Test Set: Reconstruction Error vs Ground-Truth RUL")
    st.caption("Each dot = one test engine.  Errors should be high when RUL is low.")

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    colors = ["#C62828" if ia else "#1565C0" for ia in is_anomaly]
    ax2.scatter(rul_gt, errors, c=colors, s=25, alpha=0.75)
    ax2.axhline(threshold, color="black", linestyle="--", linewidth=1.2,
                label=f"Threshold = {threshold:.4f}")
    ax2.axvline(FAULT_RUL_CUTOFF, color="gray", linestyle=":", linewidth=1,
                label=f"RUL = {FAULT_RUL_CUTOFF} label boundary")
    ax2.set_xlabel("Ground-truth RUL (cycles to failure)")
    ax2.set_ylabel("Reconstruction Error (MSE)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#C62828', markersize=7, label='Flagged as fault'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#1565C0', markersize=7, label='Classified healthy'),
    ]
    ax2.legend(handles=handles, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Tab 4: Test Results Table
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("Per-Engine Test Results")

    engine_ids = np.arange(1, len(errors) + 1)
    results_df = pd.DataFrame({
        "Engine ID"         : engine_ids,
        "Recon Error (MSE)" : np.round(errors, 6),
        "True RUL"          : rul_gt,
        "Truly Degraded"    : ["Yes" if l else "No" for l in true_labels],
        "Fault Detected"    : ["✅ YES" if a else "— no" for a in is_anomaly],
        "Status"            : [
            "True Positive"  if (t and a) else
            "True Negative"  if (not t and not a) else
            "False Positive" if (not t and a) else
            "False Negative"
            for t, a in zip(true_labels, is_anomaly)
        ]
    })

    # Colour-code the status column
    def highlight_status(val):
        colours = {
            "True Positive"  : "background-color: #c8e6c9; color: #1b5e20",
            "True Negative"  : "background-color: #e3f2fd; color: #0d47a1",
            "False Positive" : "background-color: #fff3e0; color: #e65100",
            "False Negative" : "background-color: #ffebee; color: #b71c1c",
        }
        return colours.get(val, "")

    styled = (
        results_df.style
        .applymap(highlight_status, subset=["Status"])
        .format({"Recon Error (MSE)": "{:.6f}"})
    )
    st.dataframe(styled, use_container_width=True, height=500)

    # Download button
    csv = results_df.to_csv(index=False)
    st.download_button(
        label     = "⬇ Download results as CSV",
        data      = csv,
        file_name = "fault_detection_results.csv",
        mime      = "text/csv",
    )
