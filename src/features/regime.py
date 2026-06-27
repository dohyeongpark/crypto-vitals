"""
Track A-6: Regime feature computation.

Computes basis, taker_ratio, and funding features from market_data and
UPDATEs the matching pair_features rows (same pair_id + feature_version).

Must run AFTER pair_spread (Step 5) since it updates existing rows.

Regime features (all computed from close-time data — no lookahead):
  basis_x/y        = (perp_close - spot_close) / spot_close
  taker_ratio_x/y  = taker_buy_volume / volume  (0-vol rows → NaN)
  funding_rate_x/y = perp funding_rate (already forward-filled by funding.py)
  funding_spread   = funding_rate_y - funding_rate_x

Usage:
    python -m src.features.regime
    python -m src.features.regime --version v0.2-regime
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from src.config import FEATURE_VERSION, FEATURE_WINDOW_H, INTERVAL
from src.db import get_engine, update_regime_features

logger = logging.getLogger(__name__)

PAIR_ID = "ETHUSDT_BTCUSDT"
Y_SYMBOL = "ETHUSDT"
X_SYMBOL = "BTCUSDT"


def _fetch_max_regime_ts(version: str) -> pd.Timestamp | None:
    """Return latest timestamp in pair_features that already has regime features set."""
    sql = """
        SELECT MAX(timestamp) AS max_ts
        FROM   pair_features
        WHERE  pair_id         = 'ETHUSDT_BTCUSDT'
          AND  feature_version = %(fv)s
          AND  basis_y         IS NOT NULL
    """
    df = pd.read_sql_query(sql, get_engine(), params={"fv": version})
    val = df["max_ts"].iloc[0]
    if val is None or pd.isnull(val):
        return None
    ts = pd.Timestamp(val)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def _fetch_market_data(since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load spot + perp close, volume, taker_buy_volume, funding_rate for both symbols."""
    if since is not None:
        sql = """
            SELECT symbol, market_type, timestamp,
                   close::float            AS close,
                   volume::float           AS volume,
                   taker_buy_volume::float AS taker_buy_volume,
                   funding_rate::float     AS funding_rate
            FROM   market_data
            WHERE  symbol      IN ('BTCUSDT', 'ETHUSDT')
              AND  market_type IN ('spot', 'perp')
              AND  interval    = '1h'
              AND  timestamp   >= %(since)s
            ORDER  BY timestamp
        """
        return pd.read_sql_query(sql, get_engine(), params={"since": since},
                                 parse_dates=["timestamp"])
    sql = """
        SELECT symbol, market_type, timestamp,
               close::float            AS close,
               volume::float           AS volume,
               taker_buy_volume::float AS taker_buy_volume,
               funding_rate::float     AS funding_rate
        FROM   market_data
        WHERE  symbol      IN ('BTCUSDT', 'ETHUSDT')
          AND  market_type IN ('spot', 'perp')
          AND  interval    = '1h'
        ORDER  BY timestamp
    """
    return pd.read_sql_query(sql, get_engine(), parse_dates=["timestamp"])


def _compute_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the 4-series DataFrame into a single wide frame indexed by timestamp.
    Returns columns: basis_x, basis_y, taker_ratio_x, taker_ratio_y,
                     funding_rate_x, funding_rate_y, funding_spread
    """
    def series(symbol: str, mtype: str, col: str) -> pd.Series:
        mask = (df["symbol"] == symbol) & (df["market_type"] == mtype)
        return df.loc[mask].set_index("timestamp")[col]

    btc_spot_close = series(X_SYMBOL, "spot", "close")
    btc_perp_close = series(X_SYMBOL, "perp", "close")
    btc_spot_vol   = series(X_SYMBOL, "spot", "volume")
    btc_spot_taker = series(X_SYMBOL, "spot", "taker_buy_volume")
    btc_funding    = series(X_SYMBOL, "perp", "funding_rate")

    eth_spot_close = series(Y_SYMBOL, "spot", "close")
    eth_perp_close = series(Y_SYMBOL, "perp", "close")
    eth_spot_vol   = series(Y_SYMBOL, "spot", "volume")
    eth_spot_taker = series(Y_SYMBOL, "spot", "taker_buy_volume")
    eth_funding    = series(Y_SYMBOL, "perp", "funding_rate")

    out = pd.DataFrame({
        "basis_x":        (btc_perp_close - btc_spot_close) / btc_spot_close,
        "basis_y":        (eth_perp_close - eth_spot_close) / eth_spot_close,
        "taker_ratio_x":  btc_spot_taker / btc_spot_vol.replace(0, np.nan),
        "taker_ratio_y":  eth_spot_taker / eth_spot_vol.replace(0, np.nan),
        "funding_rate_x": btc_funding,
        "funding_rate_y": eth_funding,
    })
    out["funding_spread"] = out["funding_rate_y"] - out["funding_rate_x"]
    return out


def compute_and_store(window: int, version: str, incremental: bool = False) -> int:  # noqa: ARG001 (window unused here)
    if incremental:
        max_ts = _fetch_max_regime_ts(version)
        logger.info("Incremental mode: fetching regime data from %s", max_ts)
        df_raw = _fetch_market_data(since=max_ts)
    else:
        df_raw = _fetch_market_data()
    if df_raw.empty:
        logger.error("No market_data found — run pipeline Step 1 first")
        return 0

    out = _compute_regime(df_raw)

    # Timestamps with missing spot or perp data (e.g. exchange maintenance)
    n_null = out.isnull().any(axis=1).sum()
    if n_null:
        logger.warning("%d timestamps have partial NULL regime features (expected for gap candles)", n_null)

    rows = [
        {
            "timestamp":      idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx,
            "funding_rate_y": _safe_float(row["funding_rate_y"]),
            "funding_rate_x": _safe_float(row["funding_rate_x"]),
            "funding_spread": _safe_float(row["funding_spread"]),
            "taker_ratio_y":  _safe_float(row["taker_ratio_y"]),
            "taker_ratio_x":  _safe_float(row["taker_ratio_x"]),
            "basis_y":        _safe_float(row["basis_y"]),
            "basis_x":        _safe_float(row["basis_x"]),
        }
        for idx, row in out.iterrows()
    ]

    n = update_regime_features(rows, version)
    logger.info("Updated %d pair_features rows with regime features (version=%s)", n, version)
    return n


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Compute regime features and update pair_features")
    parser.add_argument("--window", type=int, default=FEATURE_WINDOW_H)
    parser.add_argument("--version", default=FEATURE_VERSION)
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    compute_and_store(args.window, args.version, incremental=args.incremental)


if __name__ == "__main__":
    main()
