"""
train.py
--------
Training loop for the LSTM Autoencoder.

Features
--------
- Trains ONLY on healthy windows (RUL > threshold)
- Validation split from healthy data (15%)
- Early stopping — stops if val loss doesn't improve for `patience` epochs
- Saves the best model checkpoint during training
- Saves a loss-curve plot to results/training_loss.png
- Full reproducibility via torch seed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from data_loader   import load_train, load_test, download_dataset, FEATURE_COLS
from preprocessor  import prepare_training_data, save_scaler
from model         import LSTMAutoencoder, save_model, model_summary


# ─── Hyperparameters ──────────────────────────────────────────────────────────

SUBSET        = "FD001"
WINDOW_SIZE   = 30
FEATURES      = FEATURE_COLS

HIDDEN_DIM    = 64
BOTTLENECK    = 16
N_LAYERS      = 2
DROPOUT       = 0.2

BATCH_SIZE    = 64
EPOCHS        = 100
LR            = 1e-3
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.15
PATIENCE      = 10          # early stopping patience (epochs)
SEED          = 42

RESULTS_DIR   = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Training ─────────────────────────────────────────────────────────────────

def build_dataloaders(
    X_healthy: np.ndarray,
    val_split: float  = VAL_SPLIT,
    batch_size: int   = BATCH_SIZE,
    seed: int         = SEED,
) -> tuple[DataLoader, DataLoader]:
    """Split healthy windows into train/val and wrap in DataLoaders."""
    dataset = TensorDataset(torch.from_numpy(X_healthy))

    n_val   = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    print(f"[train] Train windows : {n_train:,}   Val windows : {n_val:,}")
    return train_loader, val_loader


def train_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for (x,) in loader:
        x = x.to(device)
        optimizer.zero_grad()
        recon = model(x)
        loss  = criterion(recon, x)
        loss.backward()
        # Gradient clipping — helps LSTM training stability
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: LSTMAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for (x,) in loader:
        x     = x.to(device)
        recon = model(x)
        loss  = criterion(recon, x)
        total_loss += loss.item() * len(x)
    return total_loss / len(loader.dataset)


def train(
    subset:      str   = SUBSET,
    epochs:      int   = EPOCHS,
    batch_size:  int   = BATCH_SIZE,
    lr:          float = LR,
    weight_decay:float = WEIGHT_DECAY,
    hidden_dim:  int   = HIDDEN_DIM,
    bottleneck:  int   = BOTTLENECK,
    n_layers:    int   = N_LAYERS,
    dropout:     float = DROPOUT,
    patience:    int   = PATIENCE,
    window_size: int   = WINDOW_SIZE,
    features:    list  = FEATURES,
    seed:        int   = SEED,
) -> LSTMAutoencoder:
    """
    Full training run.  Saves model and scaler to ./models/.
    Returns the best model.
    """
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"[train] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    download_dataset()
    train_df = load_train(subset)

    X_healthy, X_degraded, rul_all, scaler = prepare_training_data(
        train_df, features=features, window_size=window_size
    )
    save_scaler(scaler)

    train_loader, val_loader = build_dataloaders(X_healthy, batch_size=batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LSTMAutoencoder(
        n_features  = len(features),
        hidden_dim  = hidden_dim,
        bottleneck  = bottleneck,
        n_layers    = n_layers,
        dropout     = dropout,
    ).to(device)
    model_summary(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    train_losses, val_losses = [], []

    print(f"\n[train] Starting training for up to {epochs} epochs …\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            save_model(model)                      # save best checkpoint
        else:
            patience_count += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:>4d}/{epochs}  |  "
                f"Train MSE: {train_loss:.6f}  |  "
                f"Val MSE: {val_loss:.6f}  |  "
                f"Best: {best_val_loss:.6f}"
            )

        if patience_count >= patience:
            print(f"\n[train] Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    print(f"\n[train] Training complete.  Best val MSE: {best_val_loss:.6f}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    _plot_loss_curve(train_losses, val_losses)

    # ── Reload best model ─────────────────────────────────────────────────────
    from model import load_model
    best_model = load_model(device=str(device))
    best_model = best_model.to(device)
    return best_model


def _plot_loss_curve(train_losses: list, val_losses: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(train_losses, label="Train MSE", linewidth=1.5)
    ax.plot(val_losses,   label="Val MSE",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM Autoencoder — Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "training_loss.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[train] Loss curve saved → {out}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trained_model = train()
