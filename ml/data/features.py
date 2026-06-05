"""
Feature engineering on raw OHLCV DataFrame.

Input columns:  timestamp, open, high, low, close, volume, volatility_5m
Output columns: same + log_return, price_range, log_volume
"""

import numpy as np
import pandas as pd

# Final ordered feature columns fed into the model
FEATURE_COLS = [
    "open",
    "high",
    "low",
    "close",
    "log_return",
    "price_range",
    "log_volume",
    "volatility_5m",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to a raw OHLCV DataFrame.

    log_return   : log(close_t / close_{t-1})
                   captures per-minute return magnitude and direction.
    price_range  : (high - low) / close
                   intrabar volatility proxy, scale-independent.
    log_volume   : log1p(volume)
                   compresses volume spikes and stabilizes scale.

    The first row (NaN from the log_return shift) is dropped.
    All FEATURE_COLS are guaranteed non-null after this call.

    Returns a new DataFrame; does not modify the input.
    """
    log_return  = np.log(df["close"] / df["close"].shift(1))
    price_range = (df["high"] - df["low"]) / df["close"]
    log_volume  = np.log1p(df["volume"])

    out = (
        df.assign(
            log_return=log_return,
            price_range=price_range,
            log_volume=log_volume,
        )
        .dropna(subset=["log_return"])
        .reset_index(drop=True)
    )
    # volatility_5m is NaN for the first (window-1) rows of each contiguous segment;
    # back-fill propagates the earliest known value to those leading rows.
    out = out.assign(volatility_5m=out["volatility_5m"].bfill())
    return out
