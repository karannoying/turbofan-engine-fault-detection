"""
model.py
--------
LSTM Autoencoder for unsupervised anomaly detection on time-series sensor data.

Architecture
------------
Encoder:
    LSTM layer 1  → hidden states (captures temporal patterns)
    LSTM layer 2  → compressed bottleneck representation

Decoder:
    RepeatVector  → repeat bottleneck across the time axis
    LSTM layer 3  → reconstruct hidden states
    LSTM layer 4  → reconstruct sensor space
    Linear layer  → project back to n_features

The model is trained on healthy engine windows to minimise MSE between
input and reconstruction.  At inference time, engines in degraded states
produce higher reconstruction errors — that spike IS the anomaly signal.

        Input  →  [Encoder LSTM] → bottleneck → [Decoder LSTM] → Reconstruction
         ↑                                                               ↓
         └──────────────── MSE reconstruction loss ────────────────────┘
"""

import torch
import torch.nn as nn
from pathlib import Path


# ─── Model ────────────────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    LSTM-based sequence autoencoder.

    Parameters
    ----------
    n_features   : number of input sensor channels
    hidden_dim   : LSTM hidden state dimension
    bottleneck   : compressed latent dimension (middle LSTM hidden size)
    n_layers     : number of LSTM layers in encoder AND decoder
    dropout      : dropout probability on non-final LSTM layers
    """

    def __init__(
        self,
        n_features:  int   = 14,
        hidden_dim:  int   = 64,
        bottleneck:  int   = 16,
        n_layers:    int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.bottleneck = bottleneck
        self.n_layers   = n_layers

        # ── Encoder ──────────────────────────────────────────────────────────
        # Takes (batch, seq_len, n_features) → last hidden state (batch, bottleneck)
        self.encoder = nn.LSTM(
            input_size   = n_features,
            hidden_size  = hidden_dim,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )
        # Project to bottleneck
        self.enc_fc = nn.Linear(hidden_dim, bottleneck)

        # ── Decoder ──────────────────────────────────────────────────────────
        # Expands bottleneck back across the sequence and reconstructs sensors
        self.dec_fc = nn.Linear(bottleneck, hidden_dim)

        self.decoder = nn.LSTM(
            input_size   = hidden_dim,
            hidden_size  = hidden_dim,
            num_layers   = n_layers,
            batch_first  = True,
            dropout      = dropout if n_layers > 1 else 0.0,
        )
        # Project each timestep back to sensor space
        self.output_layer = nn.Linear(hidden_dim, n_features)


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x  : (batch, seq_len, n_features)
        out: (batch, bottleneck)
        """
        _, (h_n, _) = self.encoder(x)
        # h_n shape: (n_layers, batch, hidden_dim) — take the last layer
        last_hidden = h_n[-1]                        # (batch, hidden_dim)
        z = torch.relu(self.enc_fc(last_hidden))     # (batch, bottleneck)
        return z


    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        z      : (batch, bottleneck)
        seq_len: number of timesteps to reconstruct
        out    : (batch, seq_len, n_features)
        """
        h = torch.relu(self.dec_fc(z))               # (batch, hidden_dim)
        # Repeat latent vector across the sequence dimension
        h_seq = h.unsqueeze(1).repeat(1, seq_len, 1) # (batch, seq_len, hidden_dim)

        out, _ = self.decoder(h_seq)                 # (batch, seq_len, hidden_dim)
        recon  = self.output_layer(out)              # (batch, seq_len, n_features)
        return recon


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (batch, seq_len, n_features)
        out : (batch, seq_len, n_features)  — reconstruction
        """
        z     = self.encode(x)
        recon = self.decode(z, seq_len=x.size(1))
        return recon


    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample Mean Squared Error between input and reconstruction.

        x   : (batch, seq_len, n_features)
        out : (batch,)  — one error score per window
        """
        with torch.no_grad():
            recon = self.forward(x)
            mse   = ((x - recon) ** 2).mean(dim=(1, 2))  # mean over seq + features
        return mse


# ─── Persistence ──────────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent.parent / "models" / "lstm_autoencoder.pt"


def save_model(model: LSTMAutoencoder, path: Path = MODEL_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict" : model.state_dict(),
        "config"     : {
            "n_features" : model.n_features,
            "hidden_dim" : model.hidden_dim,
            "bottleneck" : model.bottleneck,
            "n_layers"   : model.n_layers,
        }
    }, path)
    print(f"[model] Saved → {path}")


def load_model(path: Path = MODEL_PATH, device: str = "cpu") -> LSTMAutoencoder:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LSTMAutoencoder(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


# ─── Summary helper ───────────────────────────────────────────────────────────

def model_summary(model: LSTMAutoencoder) -> None:
    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nLSTM Autoencoder")
    print(f"  Input features : {model.n_features}")
    print(f"  Hidden dim     : {model.hidden_dim}")
    print(f"  Bottleneck     : {model.bottleneck}")
    print(f"  LSTM layers    : {model.n_layers}")
    print(f"  Total params   : {total:,}")
    print(f"  Trainable      : {trainable:,}")


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = LSTMAutoencoder(n_features=14, hidden_dim=64, bottleneck=16)
    model_summary(model)

    dummy = torch.randn(32, 30, 14)          # batch=32, window=30, features=14
    recon = model(dummy)
    print(f"\nInput shape  : {dummy.shape}")
    print(f"Output shape : {recon.shape}")
    print(f"Recon errors : {model.reconstruction_error(dummy)[:5]}")
