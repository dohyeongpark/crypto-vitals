"""
Sliding-window generator for 1-minute OHLCV sequences.

For each valid window of length `lookback`:
  - Validates no timestamp gaps exist (all consecutive diffs == 60 s)
  - Applies per-window normalization
  - Computes realized volatility over the next `horizon` minutes as the label

Output shapes:
  X  : (N, lookback, n_features)  float32
  y  : (N,)                       float32  — realized volatility
  ts : (N,)                       datetime64  — last timestamp of each window
"""

import logging
import numpy as np
import pandas as pd
from .features import FEATURE_COLS

log = logging.getLogger(__name__)

_60S = pd.Timedelta(seconds=60)

# Indices within FEATURE_COLS for per-window normalization
_PRICE_IDX  = [FEATURE_COLS.index(c) for c in ("open", "high", "low", "close")]
_OTHERS_IDX = [i for i in range(len(FEATURE_COLS)) if i not in _PRICE_IDX]


def _has_gap(ts: pd.Series) -> bool:
    """Return True if any consecutive pair of timestamps differs by != 60 s."""
    return not (ts.diff().dropna() == _60S).all()


def _normalize(window: np.ndarray) -> np.ndarray:
    """
    Per-window normalization (in-place copy):

    Price columns (open/high/low/close):
        Divide by the closing price of the first bar → all values near 1.0,
        preserving relative OHLC relationships within the window.

    All other columns (log_return, price_range, log_volume, volatility_5m):
        Z-score within the window. Columns with zero std are set to 0.
    """
    out = window.copy()

    first_close = window[0, FEATURE_COLS.index("close")]
    if first_close != 0:
        for idx in _PRICE_IDX:
            out[:, idx] = window[:, idx] / first_close

    for idx in _OTHERS_IDX:
        col = window[:, idx]
        std = col.std()
        out[:, idx] = (col - col.mean()) / std if std > 0 else 0.0

    return out


def make_windows(
    df:       pd.DataFrame,
    lookback: int = 360,
    horizon:  int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build gap-free sliding windows from a feature DataFrame.

    A window [i-lookback, i) is accepted only when:
      1. All lookback consecutive bars are exactly 60 s apart.
      2. The transition from the window's last bar to the horizon's
         first bar is also exactly 60 s apart (no gap at the boundary).
      3. All horizon bars for label computation are exactly 60 s apart.

    Label = realized volatility = std(log_returns) over the next `horizon`
    minutes.  Using std of log returns (not just volatility_5m) gives an
    unbiased forward-looking target independent of the rolling window size
    already baked into volatility_5m.

    Args:
        df:       Feature DataFrame from build_features(), sorted by timestamp.
        lookback: Number of 1-minute bars per input window (default 360 = 6 h).
        horizon:  Number of future bars used for the realized-vol label (default 30).

    Returns:
        X:  float32 array of shape (N, lookback, n_features)
        y:  float32 array of shape (N,)
        ts: datetime64 array of shape (N,) — last timestamp of each window
    """
    values     = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy()
    closes     = df["close"].to_numpy(dtype=np.float64)
    n          = len(df)

    X_list, y_list, ts_list = [], [], []
    skipped_gap = 0

    for i in range(lookback, n - horizon):
        win_ts = df["timestamp"].iloc[i - lookback : i]
        if _has_gap(win_ts):
            skipped_gap += 1
            continue

        # Also verify no gap between window end and future horizon
        boundary_ts = df["timestamp"].iloc[i - 1 : i + horizon]
        if _has_gap(boundary_ts):
            skipped_gap += 1
            continue

        # Realized volatility label: std of log returns over [i, i+horizon)
        # Need close[i-1] as the base for the first future return
        fut_closes      = closes[i - 1 : i + horizon]          # horizon + 1 values
        fut_log_returns = np.log(fut_closes[1:] / fut_closes[:-1])
        realized_vol    = float(np.std(fut_log_returns))

        X_list.append(_normalize(values[i - lookback : i]))
        y_list.append(realized_vol)
        ts_list.append(timestamps[i - 1])

    if not X_list:
        raise ValueError(
            f"No valid windows found for lookback={lookback}, horizon={horizon}. "
            "Check for data gaps or insufficient rows."
        )

    log.info(
        "Windows: %d valid, %d skipped (gap)",
        len(X_list), skipped_gap,
    )

    return (
        np.stack(X_list, axis=0).astype(np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(ts_list),
    )
