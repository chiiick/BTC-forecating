"""Evaluation metrics and a simple sign-based backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models import build_model
from preprocess import TARGETS, build_data_bundle

HORIZON_NAMES = ["2h", "4h", "6h"]


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.0) -> float:
    mask = np.ones_like(y_pred, dtype=bool) if eps == 0 else np.abs(y_pred) >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(((y_true[mask] >= 0) == (y_pred[mask] >= 0)).mean())


def metrics_multi(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, horizon in enumerate(HORIZON_NAMES):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        out[f"{horizon}_MAE"] = float(mean_absolute_error(yt, yp))
        out[f"{horizon}_RMSE"] = float(np.sqrt(mean_squared_error(yt, yp)))
        out[f"{horizon}_DA"] = direction_accuracy(yt, yp, eps=0.0)
        out[f"{horizon}_DAe"] = direction_accuracy(yt, yp, eps=1e-4)
    return out


def evaluate_model(model, dataloader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    n = 0
    ys: list[np.ndarray] = []
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            batch_size = len(y)
            total_loss += float(loss.item()) * batch_size
            n += batch_size
            ys.append(y.cpu().numpy())
            preds.append(pred.cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    return total_loss / max(n, 1), metrics_multi(y_true, y_pred), y_true, y_pred


def max_drawdown(cumulative_return: np.ndarray) -> float:
    x = np.asarray(cumulative_return, dtype=float)
    if x.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(x)
    return float((x - peak).min())


def backtest_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 0.0,
    transaction_cost_bps: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Simple long/short diagnostic backtest used in the original project.

    This is intentionally lightweight: one unit, sign(prediction), no position sizing.
    """
    results: dict[str, dict[str, float]] = {}

    for i, horizon in enumerate(HORIZON_NAMES):
        yt = np.asarray(y_true[:, i], dtype=float)
        yp = np.asarray(y_pred[:, i], dtype=float)
        signal = np.where(np.abs(yp) >= eps, np.sign(yp), 0.0)

        trades = int(np.sum(np.abs(np.diff(signal)) > 0) + (signal[0] != 0)) if len(signal) else 0
        total_cost = trades * (transaction_cost_bps / 10000.0)
        strategy_return = signal * yt

        cumulative_return = float(np.nansum(strategy_return) - total_cost)
        mean_return = float(np.nanmean(strategy_return))
        std_return = float(np.nanstd(strategy_return, ddof=1))
        sharpe_like = mean_return / std_return if std_return > 0 else float("nan")
        equity = np.nancumsum(strategy_return) - np.linspace(0, total_cost, num=len(strategy_return))

        results[horizon] = {
            "RET": cumulative_return,
            "SR": float(sharpe_like),
            "MDD": max_drawdown(equity),
            "Hit": direction_accuracy(yt, yp),
            "Trades": float(trades),
        }

    return results


def backtest_to_frame(backtests: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    rows = []
    for model_name, model_results in backtests.items():
        for horizon, metrics in model_results.items():
            rows.append({"model": model_name, "horizon": horizon, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["gru", "lstm", "tcn", "transformer"], required=True)
    parser.add_argument("--data", type=Path, default=repo_root / "data" / "btc_2y_1h.csv")
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_data_bundle(args.data, window=args.window, batch_size=args.batch_size)
    model = build_model(args.model, bundle.in_features).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict)

    criterion = torch.nn.HuberLoss(delta=1.0)
    loss, metrics, y_true, y_pred = evaluate_model(model, bundle.test_loader, criterion, device)
    backtest = backtest_from_predictions(y_true, y_pred)

    print(f"test_loss={loss:.8f}")
    print(pd.Series(metrics).to_string())
    print(pd.DataFrame(backtest).T.to_string())


if __name__ == "__main__":
    main()
