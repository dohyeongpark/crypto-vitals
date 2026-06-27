"""
Track A-5: Pair spread and z-score computation (ETH/BTC).

Computes: log prices, rolling OLS β, spread, rolling mean/std, robust z-score (MAD).
Inserts into pair_features.

Robust z-score: (spread - rolling_median) / (MAD * 1.4826)
The 1.4826 constant makes MAD consistent with normal σ.

Usage:
    python -m src.features.pair_spread
    python -m src.features.pair_spread --window 168 --version v0.1-ols
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

import pandas as pd

from src.config import FEATURE_VERSION, FEATURE_WINDOW_H, INTERVAL
from src.db import get_engine, upsert_pair_features

logger = logging.getLogger(__name__)

PAIR_ID = "ETHUSDT_BTCUSDT"
Y_SYMBOL = "ETHUSDT"  # y (dependent)
X_SYMBOL = "BTCUSDT"  # x (independent)


def _fetch_max_processed_ts(version: str) -> pd.Timestamp | None:
    """Return the latest timestamp already in pair_features for this version."""
    sql = """
        SELECT MAX(timestamp) AS max_ts
        FROM   pair_features
        WHERE  pair_id         = 'ETHUSDT_BTCUSDT'
          AND  feature_version = %(fv)s
    """
    df = pd.read_sql_query(sql, get_engine(), params={"fv": version})
    val = df["max_ts"].iloc[0]
    if val is None or (hasattr(val, "isnull") and val.isnull()):
        return None
    ts = pd.Timestamp(val)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def _fetch_spot_closes(since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load spot close prices for BTC and ETH, aligned by timestamp."""
    if since is not None:
        sql = """
            SELECT timestamp, symbol, close
            FROM   market_data
            WHERE  symbol      IN ('BTCUSDT', 'ETHUSDT')
              AND  market_type = 'spot'
              AND  interval    = '1h'
              AND  timestamp   >= %(since)s
            ORDER  BY timestamp
        """
        df = pd.read_sql_query(sql, get_engine(), params={"since": since},
                               parse_dates=["timestamp"])
    else:
        sql = """
            SELECT timestamp, symbol, close
            FROM   market_data
            WHERE  symbol      IN ('BTCUSDT', 'ETHUSDT')
              AND  market_type = 'spot'
              AND  interval    = '1h'
            ORDER  BY timestamp
        """
        df = pd.read_sql_query(sql, get_engine(), parse_dates=["timestamp"])

    # Pivot to wide format: columns = [BTCUSDT, ETHUSDT]
    wide = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    wide = wide.dropna()  # only aligned timestamps
    wide.index = wide.index.tz_localize("UTC") if wide.index.tz is None else wide.index
    return wide


def _rolling_ols_beta(log_y: pd.Series, log_x: pd.Series, window: int) -> pd.Series:
    """
    Rolling OLS β = cov(y, x) / var(x) over `window` periods.
    Computed without shift — result at t uses data [t-window+1, t].
    No future data is accessed.
    """
    cov = log_y.rolling(window).cov(log_x)
    var = log_x.rolling(window).var()
    return cov / var


def _robust_zscore(spread: pd.Series, window: int) -> pd.Series:
    """
    Rolling robust z-score via MAD.
    z = (x - median) / (MAD * 1.4826)
    """
    rolling_median = spread.rolling(window).median()
    rolling_mad = spread.rolling(window).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    )
    return (spread - rolling_median) / (rolling_mad * 1.4826)


def compute_and_store(window: int, version: str, incremental: bool = False) -> int:
    if incremental:
        max_ts = _fetch_max_processed_ts(version)
        # Fetch with 2× window overlap so rolling stats are valid at the boundary
        fetch_since = (max_ts - pd.Timedelta(hours=window * 2)) if max_ts else None
        logger.info("Incremental mode: fetching from %s (max_processed=%s)", fetch_since, max_ts)
    else:
        fetch_since = None

    wide = _fetch_spot_closes(since=fetch_since)
    if wide.empty or Y_SYMBOL not in wide.columns or X_SYMBOL not in wide.columns:
        logger.error("Missing close data for %s or %s", Y_SYMBOL, X_SYMBOL)
        return 0

    log_y = np.log(wide[Y_SYMBOL].astype(float))
    log_x = np.log(wide[X_SYMBOL].astype(float))

    beta = _rolling_ols_beta(log_y, log_x, window)
    spread = log_y - beta * log_x

    spread_mean = spread.rolling(window).mean()
    spread_std = spread.rolling(window).std()
    zscore = _robust_zscore(spread, window)

    result = pd.DataFrame({
        "log_price_y": log_y,
        "log_price_x": log_x,
        "hedge_beta": beta,
        "spread": spread,
        "spread_mean": spread_mean,
        "spread_std": spread_std,
        "zscore": zscore,
    }).dropna()

    if result.empty:
        logger.warning("No valid rows after rolling computation (window=%d)", window)
        return 0

    rows = [
        {
            "pair_id": PAIR_ID,
            "interval": INTERVAL,
            "timestamp": idx.to_pydatetime(),
            "log_price_y": float(row["log_price_y"]),
            "log_price_x": float(row["log_price_x"]),
            "hedge_beta": float(row["hedge_beta"]),
            "spread": float(row["spread"]),
            "spread_mean": float(row["spread_mean"]),
            "spread_std": float(row["spread_std"]),
            "zscore": float(row["zscore"]),
            "feature_version": version,
        }
        for idx, row in result.iterrows()
    ]

    n = upsert_pair_features(rows)
    logger.info("Inserted %d pair_features rows (window=%d, version=%s)", n, window, version)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Compute ETH/BTC pair spread and z-score")
    parser.add_argument("--window", type=int, default=FEATURE_WINDOW_H, help="Rolling window in hours")
    parser.add_argument("--version", default=FEATURE_VERSION, help="feature_version tag")
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    compute_and_store(args.window, args.version, incremental=args.incremental)


if __name__ == "__main__":
    main()
