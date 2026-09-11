"""Download BTC/USDT 1-hour OHLCV data from Binance via ccxt.

This is a cleaned version of the original BTC.py used for the course project.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ccxt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def download_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start: str = "2023-10-15T00:00:00Z",
    end: str = "2025-10-15T23:59:59Z",
    limit: int = 1000,
) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    since_ms = exchange.parse8601(start)
    end_ms = exchange.parse8601(end)

    bars: list[list[float]] = []
    cursor = since_ms

    while cursor <= end_ms:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=cursor,
            limit=limit,
        )
        if not batch:
            break

        bars.extend(batch)
        next_cursor = int(batch[-1][0]) + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor

    df = pd.DataFrame(
        bars,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    if df.empty:
        raise RuntimeError("No OHLCV data was returned by Binance.")

    # ccxt can return a full final batch beyond the requested end time.
    df = df[(df["timestamp"] >= since_ms) & (df["timestamp"] <= end_ms)].copy()
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    return df


def save_eda_plots(df: pd.DataFrame, assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 3))
    plt.plot(df["time"], df["close"])
    plt.title("BTC/USDT Close Price (1h, 2023-2025)")
    plt.tight_layout()
    plt.savefig(assets_dir / "btc_close.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 3))
    plt.plot(df["time"], df["volume"])
    plt.title("BTC/USDT Volume (1h, 2023-2025)")
    plt.tight_layout()
    plt.savefig(assets_dir / "btc_volume.png", dpi=150)
    plt.close()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--start", default="2023-10-15T00:00:00Z")
    parser.add_argument("--end", default="2025-10-15T23:59:59Z")
    parser.add_argument("--output", type=Path, default=repo_root / "data" / "btc_2y_1h.csv")
    parser.add_argument("--assets-dir", type=Path, default=repo_root / "assets")
    args = parser.parse_args()

    df = download_ohlcv(args.symbol, args.timeframe, args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    save_eda_plots(df, args.assets_dir)

    print(f"Saved {len(df):,} rows to {args.output}")
    print(f"Period: {df['time'].min()} -> {df['time'].max()}")


if __name__ == "__main__":
    main()
