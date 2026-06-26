"""
Track A-4: Realized volatility computation.

Computes rolling realized volatility from log returns and UPDATEs market_data.
realized_vol = rolling std of log(close_t / close_{t-1}) over `window` periods.

Usage:
    python -m src.features.volatility
    python -m src.features.volatility --window 24 --symbols BTCUSDT --market spot
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from src.config import FEATURE_WINDOW_H, INTERVAL, SYMBOLS
from src.db import get_conn, get_engine, update_realized_vol

logger = logging.getLogger(__name__)


def _fetch_closes(symbol: str, market_type: str) -> pd.DataFrame:
    sql = """
        SELECT timestamp, close
        FROM   market_data
        WHERE  symbol      = %(symbol)s
          AND  market_type = %(market_type)s
          AND  interval    = '1h'
        ORDER  BY timestamp
    """
    return pd.read_sql_query(
        sql, get_engine(),
        params={"symbol": symbol, "market_type": market_type},
        parse_dates=["timestamp"],
    )


def compute_and_store(symbol: str, market_type: str, window: int) -> int:
    df = _fetch_closes(symbol, market_type)
    if df.empty:
        logger.warning("[%s/%s] No data, skipping realized vol", symbol, market_type)
        return 0

    df["log_ret"] = np.log(df["close"].astype(float) / df["close"].astype(float).shift(1))
    df["realized_vol"] = df["log_ret"].rolling(window).std()

    # Only write rows where we have a valid vol value
    valid = df.dropna(subset=["realized_vol"])[["timestamp", "realized_vol"]].copy()
    if valid.empty:
        logger.info("[%s/%s] Not enough rows for window=%d", symbol, market_type, window)
        return 0

    rows = [
        {
            "symbol": symbol,
            "market_type": market_type,
            "interval": INTERVAL,
            "timestamp": row["timestamp"].to_pydatetime(),
            "realized_vol": float(row["realized_vol"]),
        }
        for _, row in valid.iterrows()
    ]

    n = update_realized_vol(rows)
    logger.info("[%s/%s] updated %d realized_vol rows (window=%d)", symbol, market_type, n, window)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Compute realized volatility and write to market_data")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--market", nargs="+", default=["spot", "perp"], choices=["spot", "perp"])
    parser.add_argument("--window", type=int, default=FEATURE_WINDOW_H, help="Rolling window in hours (default: FEATURE_WINDOW_H)")
    args = parser.parse_args()

    for symbol in args.symbols:
        for market_type in args.market:
            compute_and_store(symbol, market_type, args.window)


if __name__ == "__main__":
    main()
