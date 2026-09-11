"""Preprocessing utilities for multi-horizon BTC/USDT forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import ta
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

TARGETS = ["y_2h", "y_4h", "y_6h"]
HORIZONS = [2, 4, 6]

# Explicitly listed feature set from the final project presentation.
FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_sig",
    "atr_14",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


class TimeSeriesDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    scaler: StandardScaler
    feature_cols: list[str]
    targets: list[str]
    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    @property
    def in_features(self) -> int:
        return self.x_train.shape[-1]


def _normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "time" in df.columns:
        raw_time = df["time"]
    elif "timestamp" in df.columns:
        raw_time = df["timestamp"]
    else:
        raise ValueError("CSV must contain either 'time' or 'timestamp'.")

    if pd.api.types.is_numeric_dtype(raw_time):
        max_value = float(pd.to_numeric(raw_time, errors="coerce").max())
        unit = "ms" if max_value > 1e12 else "s"
        df["time"] = pd.to_datetime(raw_time, unit=unit, utc=True, errors="coerce")
    else:
        df["time"] = pd.to_datetime(raw_time, utc=True, errors="coerce")

    return df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12, fillna=False)
    df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26, fillna=False)
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14, fillna=False)

    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_sig"] = macd.macd_signal()
    df["atr_14"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14, fillna=False
    )

    hour = df["time"].dt.hour.astype(float)
    dow = df["time"].dt.dayofweek.astype(float)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return df


def add_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add correctly aligned cumulative future log-return targets.

    y_k(t) = r(t+1) + ... + r(t+k), for k in {2, 4, 6}.
    """
    df = df.copy()
    for horizon in HORIZONS:
        df[f"y_{horizon}h"] = sum(
            df["log_return"].shift(-step) for step in range(1, horizon + 1)
        )
    return df


def load_dataframe(
    csv_path: str | Path,
    start: str = "2023-10-15T00:00:00Z",
    end: str = "2025-10-15T23:59:59Z",
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df.columns = [str(c).lower() for c in df.columns]
    df = _normalize_time(df)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")

    df = df[(df["time"] >= start_ts) & (df["time"] <= end_ts)].copy()
    df = add_features(df)
    df = add_future_targets(df)
    df = df.dropna(subset=FEATURES + TARGETS).reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    train_end: str = "2025-05-31T23:59:59Z",
    val_end: str = "2025-08-31T23:59:59Z",
    test_end: str = "2025-10-15T23:59:59Z",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time while ensuring future targets stay inside each split."""
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)
    test_end_ts = pd.Timestamp(test_end)
    max_h = pd.Timedelta(hours=max(HORIZONS))

    train = df[df["time"] <= train_end_ts - max_h].copy()
    val = df[(df["time"] > train_end_ts) & (df["time"] <= val_end_ts - max_h)].copy()
    test = df[(df["time"] > val_end_ts) & (df["time"] <= test_end_ts - max_h)].copy()

    if min(len(train), len(val), len(test)) == 0:
        raise ValueError("One of the chronological splits is empty.")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_windows(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_cols: list[str],
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_scaled = scaler.transform(frame[feature_cols].to_numpy(dtype=np.float64))
    y_values = frame[TARGETS].to_numpy(dtype=np.float64)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    # The input ends at time t, and y(t) contains t+1 ... t+k returns.
    for t in range(window - 1, len(frame)):
        xs.append(x_scaled[t - window + 1 : t + 1])
        ys.append(y_values[t])

    if not xs:
        raise ValueError(f"Split has fewer rows ({len(frame)}) than window ({window}).")

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def build_data_bundle(
    csv_path: str | Path,
    window: int = 240,
    batch_size: int = 256,
    num_workers: int = 0,
) -> DataBundle:
    df = load_dataframe(csv_path)
    train_frame, val_frame, test_frame = chronological_split(df)

    scaler = StandardScaler().fit(train_frame[FEATURES].to_numpy(dtype=np.float64))

    x_train, y_train = make_windows(train_frame, scaler, FEATURES, window)
    x_val, y_val = make_windows(val_frame, scaler, FEATURES, window)
    x_test, y_test = make_windows(test_frame, scaler, FEATURES, window)

    train_ds = TimeSeriesDataset(x_train, y_train)
    val_ds = TimeSeriesDataset(x_val, y_val)
    test_ds = TimeSeriesDataset(x_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scaler=scaler,
        feature_cols=FEATURES.copy(),
        targets=TARGETS.copy(),
        train_frame=train_frame,
        val_frame=val_frame,
        test_frame=test_frame,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )
