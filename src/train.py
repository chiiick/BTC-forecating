"""Train GRU/LSTM/TCN/Transformer models for 2h/4h/6h BTC return forecasting."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluate import backtest_from_predictions, backtest_to_frame, evaluate_model
from models import build_model
from preprocess import build_data_bundle


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    checkpoint_path: Path,
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 6,
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    wait = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        running_loss = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += float(loss.item()) * len(y)
            n += len(y)

        train_loss = running_loss / max(n, 1)
        val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        da_text = " ".join(
            f"{h}_DA={val_metrics[f'{h}_DA'] * 100:.2f}%" for h in ["2h", "4h", "6h"]
        )
        print(
            f"epoch={epoch:02d} train={train_loss:.8f} val={val_loss:.8f} "
            f"{da_text} time={time.time() - started:.1f}s"
        )

        if val_loss < best_val - 1e-9:
            best_val = val_loss
            wait = 0
            torch.save({"state_dict": model.state_dict()}, checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_metrics, y_true, y_pred = evaluate_model(model, test_loader, criterion, device)
    return model, test_loss, test_metrics, y_true, y_pred


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=repo_root / "data" / "btc_2y_1h.csv")
    parser.add_argument("--models", nargs="+", default=["gru", "lstm", "tcn", "transformer"],
                        choices=["gru", "lstm", "tcn", "transformer"])
    parser.add_argument("--window", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    bundle = build_data_bundle(args.data, window=args.window, batch_size=args.batch_size)
    print(
        "samples:",
        f"train={len(bundle.x_train):,}",
        f"val={len(bundle.x_val):,}",
        f"test={len(bundle.x_test):,}",
        f"features={bundle.in_features}",
    )

    checkpoints_dir = repo_root / "artifacts" / "checkpoints"
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    backtests = {}

    for model_name in args.models:
        print(f"\n=== {model_name.upper()} ===")
        model = build_model(model_name, bundle.in_features)
        checkpoint_path = checkpoints_dir / f"{model_name}.pt"
        model, test_loss, metrics, y_true, y_pred = train_one(
            model,
            bundle.train_loader,
            bundle.val_loader,
            bundle.test_loader,
            device,
            checkpoint_path,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
        )

        row = {"model": model_name, "test_loss": test_loss, **metrics}
        metric_rows.append(row)
        backtests[model_name] = backtest_from_predictions(
            y_true,
            y_pred,
            transaction_cost_bps=args.transaction_cost_bps,
        )
        np.savez_compressed(
            results_dir / f"predictions_{model_name}.npz",
            y_true=y_true,
            y_pred=y_pred,
        )

    metrics_df = pd.DataFrame(metric_rows)
    backtest_df = backtest_to_frame(backtests)
    metrics_df.to_csv(results_dir / "test_metrics.csv", index=False)
    backtest_df.to_csv(results_dir / "backtest_metrics.csv", index=False)

    run_config = {
        "models": args.models,
        "window": args.window,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "seed": args.seed,
        "transaction_cost_bps": args.transaction_cost_bps,
        "features": bundle.feature_cols,
        "targets": bundle.targets,
    }
    (results_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print("\n=== TEST METRICS ===")
    print(metrics_df.to_string(index=False))
    print("\n=== BACKTEST METRICS ===")
    print(backtest_df.to_string(index=False))


if __name__ == "__main__":
    main()
