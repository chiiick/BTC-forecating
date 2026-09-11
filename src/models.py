"""Model definitions: GRU, LSTM, TCN, and a lightweight Transformer encoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class RNNMulti(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int = 64,
        layers: int = 2,
        kind: str = "gru",
        dropout: float = 0.2,
        out_dim: int = 3,
    ):
        super().__init__()
        kind = kind.lower()
        if kind not in {"gru", "lstm"}:
            raise ValueError("kind must be 'gru' or 'lstm'.")

        rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            in_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        padding = (kernel_size - 1) * dilation
        super().__init__(
            in_ch,
            out_ch,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.left_trim = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return out[..., :-self.left_trim] if self.left_trim > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.residual(x)


class TCNMulti(nn.Module):
    def __init__(
        self,
        in_features: int,
        channels: int = 64,
        depth: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        out_dim: int = 3,
    ):
        super().__init__()
        blocks = []
        in_ch = in_features
        for i in range(depth):
            blocks.append(
                TCNBlock(
                    in_ch,
                    channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            in_ch = channels

        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, W, F) -> (B, F, W)
        return self.head(self.tcn(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerMulti(nn.Module):
    def __init__(
        self,
        in_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        out_dim: int = 3,
    ):
        super().__init__()
        self.input_projection = nn.Linear(in_features, d_model)
        self.position = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.position(self.input_projection(x))
        z = self.encoder(z)
        return self.head(z[:, -1, :])


def build_model(name: str, in_features: int) -> nn.Module:
    name = name.lower()
    if name == "gru":
        return RNNMulti(in_features, kind="gru")
    if name == "lstm":
        return RNNMulti(in_features, kind="lstm")
    if name == "tcn":
        return TCNMulti(in_features)
    if name in {"transformer", "trans"}:
        return TransformerMulti(in_features)
    raise ValueError(f"Unknown model: {name}")
