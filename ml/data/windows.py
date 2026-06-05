"""
Valid-index computation for on-demand sliding windows.

Instead of materializing all windows up front (which causes OOM for 1 year
of 1-minute data), this module only computes *which* positions in the
DataFrame are valid window endpoints.  The actual slicing, normalization,
and label computation happen lazily inside VolatilityDataset.__getitem__.

Memory comparison (1 year, lookback=360, 8 features):
  Pre-materialized  526 K × 360 × 8 × 4 B  ≈ 6.1 GB
  Index-only        526 K × 4 B             ≈ 2 MB
"""

import logging
import numpy as np
import pandas as pd
from .features import FEATURE_COLS

log = logging.getLogger(__name__)

_60S = pd.Timedelta(seconds=60)

# Column positions used by the normalizer
_CLOSE_IDX  = FEATURE_COLS.index("close")
_PRICE_IDX  = [FEATURE_COLS.index(c) for c in ("open", "high", "low", "close")]
_OTHERS_IDX = [i for i in range(len(FEATURE_COLS)) if i not in _PRICE_IDX]


def normalize(window: np.ndarray) -> np.ndarray:
    """
    Per-window normalization.

    Price columns (open / high / low / close):
        Divided by the first bar's close → values near 1.0,
        relative OHLC relationships preserved.

    All other columns (log_return, price_range, log_volume, volatility_5m):
        Z-scored within the window.  Columns with std == 0 are set to 0.

    Args:
        window: float32 array of shape (lookback, n_features)
    Returns:
        Normalized copy, same shape.
    """
    out = window.copy()
    first_close = window[0, _CLOSE_IDX]
    if first_close != 0:
        for idx in _PRICE_IDX:
            out[:, idx] = window[:, idx] / first_close
    for idx in _OTHERS_IDX:
        col = window[:, idx]
        std = col.std()
        out[:, idx] = (col - col.mean()) / std if std > 0 else 0.0
    return out


def get_valid_indices(
    df:       pd.DataFrame,
    lookback: int = 360,
    horizon:  int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scan the DataFrame and return the indices of valid window endpoints.

    A position i (0-based) is valid when:
      1. df[i-lookback : i]  — all consecutive timestamps are 60 s apart.
      2. df[i-1 : i+horizon] — no gap at the window/future boundary either.

    Args:
        df:       Feature DataFrame from build_features(), sorted by timestamp.
        lookback: Input window length in minutes (default 360 = 6 h).
        horizon:  Future window length used for label computation (default 30).

    Returns:
        indices : int64 array of valid i values  (window covers df[i-lookback:i])
        ts      : datetime64 array — timestamp of the last bar in each window
    """
    timestamps = df["timestamp"].to_numpy()
    n          = len(df)
    valid_idx  = []
    skipped    = 0

    for i in range(lookback, n - horizon):
        # --- gap check: lookback window ---
        win_ts = df["timestamp"].iloc[i - lookback : i]
        if not (win_ts.diff().dropna() == _60S).all():
            skipped += 1
            continue

        # --- gap check: window-to-horizon boundary ---
        boundary_ts = df["timestamp"].iloc[i - 1 : i + horizon]
        if not (boundary_ts.diff().dropna() == _60S).all():
            skipped += 1
            continue

        valid_idx.append(i)

    indices = np.array(valid_idx, dtype=np.int64)
    ts      = timestamps[indices - 1]   # last timestamp of each window

    log.info(
        "Valid indices: %d  |  skipped (gap): %d  |  lookback=%d  horizon=%d",
        len(indices), skipped, lookback, horizon,
    )
    return indices, ts
