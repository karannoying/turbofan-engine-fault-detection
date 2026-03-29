# ✈ Turbofan Engine Fault Detection
### LSTM Autoencoder · NASA CMAPSS · Unsupervised Anomaly Detection

A real-world aerospace ML project that detects engine degradation from sensor
data — the same approach used in industrial Prognostics and Health Management
(PHM) systems at GE Aviation and Rolls-Royce.

---

## Project Structure

```
turbofan_fault_detection/
├── run.py                   ← Master script: runs the full pipeline
├── app.py                   ← Streamlit interactive dashboard
├── requirements.txt
├── src/
│   ├── data_loader.py       ← Downloads & parses NASA CMAPSS dataset
│   ├── preprocessor.py      ← Normalisation, smoothing, sliding windows
│   ├── model.py             ← LSTM Autoencoder architecture
│   ├── train.py             ← Training loop with early stopping
│   ├── detect.py            ← Reconstruction error scoring & threshold
│   └── evaluate.py          ← Metrics, confusion matrix, ROC, plots
├── data/                    ← CMAPSS .txt files (auto-downloaded)
├── models/                  ← Saved model + scaler + threshold
└── results/                 ← Generated plots (PNG)
```

---

## Quick Start

```bash
# 1. Clone / navigate to the project folder
cd turbofan_fault_detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (downloads data, trains, evaluates)
python run.py

# 4. Launch the interactive dashboard
streamlit run app.py
```

That's it.  The pipeline will:
- Auto-download the NASA CMAPSS FD001 dataset
- Preprocess 21 sensor channels into 30-cycle sliding windows
- Train the LSTM Autoencoder for up to 100 epochs (early stopping)
- Calibrate an anomaly threshold at the 95th percentile of healthy errors
- Evaluate against ground-truth RUL labels
- Save all result plots to `./results/`

---

## How It Works

### The Core Idea

```
Normal engine  →  Autoencoder  →  Low reconstruction error   ✅
Degraded engine →  Autoencoder  →  HIGH reconstruction error  🚨 FAULT
```

The LSTM Autoencoder is trained **exclusively on healthy engine cycles** (the
first ~60% of each engine's life, where RUL > 125 cycles).  It learns to
compress and reconstruct normal multi-variate sensor patterns.

When a degraded engine's sensor readings deviate from the learnt normal
distribution, the model cannot reconstruct them accurately — reconstruction
error spikes, triggering an alert.

### Architecture

```
Input (batch, 30, 14)
    ↓
[Encoder LSTM ×2]     64 hidden units → 16 bottleneck
    ↓
Bottleneck (batch, 16)
    ↓
[Decoder LSTM ×2]     16 → 64 hidden units
    ↓
Linear projection
    ↓
Reconstruction (batch, 30, 14)
    ↓
MSE(input, reconstruction) → anomaly score
```

### Dataset

**NASA CMAPSS (Commercial Modular Aero-Propulsion System Simulation)**
- 4 sub-datasets (FD001 – FD004), each with different operating conditions & fault modes
- FD001: ~100 engines, 1 operating condition, 1 fault mode → simplest, start here
- 21 sensor channels: temperatures, pressures, fan speeds, fuel flow, etc.
- Each engine runs until failure; ground-truth RUL provided for the test set
- Download: https://data.nasa.gov/dataset/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

### Features Used (14 of 21 sensors)

Sensors with near-zero variance across all engines in FD001 are dropped
(s1, s5, s6, s10, s16, s18, s19) — they carry no useful signal.

Remaining: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21
plus 3 operational settings (op1, op2, op3).

---

## CLI Options

```bash
python run.py --help

  --subset          FD001 | FD002 | FD003 | FD004  (default: FD001)
  --epochs          Max training epochs             (default: 100)
  --batch-size      Training batch size             (default: 64)
  --lr              Learning rate                   (default: 0.001)
  --window-size     Sliding window length (cycles)  (default: 30)
  --threshold-percentile   Anomaly threshold pct    (default: 95.0)
  --skip-training   Load existing model, skip train
```

---

## Expected Results (FD001)

| Metric    | Typical Range |
|-----------|---------------|
| Precision | 0.75 – 0.90   |
| Recall    | 0.65 – 0.85   |
| F1-Score  | 0.70 – 0.87   |
| ROC-AUC   | 0.80 – 0.92   |

*Results vary with random seed, threshold percentile, and subset chosen.*

---

## Extending the Project

- **Try FD002/FD004** — multiple operating conditions make it much harder; add
  a k-means operating-regime classifier first, then train one autoencoder per regime.
- **Add SHAP explanations** — use `shap` library to show which sensors contribute
  most to the anomaly score.
- **Swap to a Transformer** — replace the LSTM layers with a Transformer encoder
  for potentially better long-range temporal modelling.
- **RUL regression** — instead of binary fault/no-fault, add a regression head
  to the bottleneck to predict remaining useful life directly.
- **Real-time streaming simulation** — feed cycles one at a time and plot the
  live anomaly score.

---

## References

- Saxena & Goebel (2008). *Turbofan Engine Degradation Simulation Data Set.*
  NASA Ames Prognostics Data Repository.
- Malhotra et al. (2016). *LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection.*
  ICML Anomaly Detection Workshop.
