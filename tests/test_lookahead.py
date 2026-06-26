"""
Tests to verify there is no lookahead bias in the data pipeline.

Rule: for a candle with timestamp T (= close time), all features computed
at T must use only data with timestamp ≤ T. Open time is never used as the
stored timestamp (that would shift everything 1h into the future).

These tests operate on synthetic / parsed data and do NOT require a live DB.
"""
import io
import csv
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. CSV parsing: timestamp == close_time, NOT open_time
# ─────────────────────────────────────────────────────────────────────────────

def _make_csv_zip(open_time_ms: int, close_time_ms: int) -> bytes:
    """Create a minimal Binance Vision klines ZIP in memory."""
    row = [
        str(open_time_ms), "30000", "31000", "29000", "30500", "100",
        str(close_time_ms), "3050000", "500", "50", "1525000", "0",
    ]
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(row)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("BTCUSDT-1h-2024-01.csv", buf.getvalue())
    zip_buf.seek(0)
    return zip_buf.read()


def test_timestamp_is_close_time_not_open_time():
    """Parsed timestamp must equal close_time rounded to hour boundary, not open_time."""
    from src.collectors.klines_bulk import _parse_csv

    # 2024-01-01 00:00:00 UTC open → 2024-01-01 00:59:59.999 UTC close
    open_ms = int(datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    # close_time_ms = open_ms + 3_599_999 (Binance convention for 1h)
    close_ms = open_ms + 3_599_999

    raw_content = _make_csv_zip(open_ms, close_ms)
    with zipfile.ZipFile(io.BytesIO(raw_content)) as zf:
        csv_data = zf.read(zf.namelist()[0])

    rows = _parse_csv(csv_data, "BTCUSDT", "spot")
    assert len(rows) == 1

    ts: datetime = rows[0]["timestamp"]
    expected_close = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)  # close+1ms rounds to next hour

    assert ts == expected_close, (
        f"timestamp should be close_time ({expected_close}), got {ts}. "
        "Using open_time would introduce 1h lookahead bias."
    )


def test_open_time_not_stored():
    """Confirm stored timestamp is NOT the open_time."""
    from src.collectors.klines_bulk import _parse_csv

    open_ms = int(datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    close_ms = open_ms + 3_599_999

    raw_content = _make_csv_zip(open_ms, close_ms)
    with zipfile.ZipFile(io.BytesIO(raw_content)) as zf:
        csv_data = zf.read(zf.namelist()[0])

    rows = _parse_csv(csv_data, "ETHUSDT", "spot")
    ts: datetime = rows[0]["timestamp"]

    open_time = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
    assert ts != open_time, "timestamp must NOT equal open_time (lookahead)"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Rolling β: no future data accessed
# ─────────────────────────────────────────────────────────────────────────────

def _make_price_df(n: int = 500, window: int = 168) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    times = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    btc = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 40000, index=times)
    eth = pd.Series(np.cumsum(rng.normal(0, 0.5, n)) + 2000, index=times)
    return pd.DataFrame({"BTCUSDT": btc, "ETHUSDT": eth})


def test_rolling_beta_no_future_data():
    """
    Rolling OLS β at index i must only use data from [i-window+1, i].
    Verify by computing β on a truncated series and confirming it matches
    the full-series value at the same position.
    """
    from src.features.pair_spread import _rolling_ols_beta

    window = 24
    df = _make_price_df(n=200, window=window)
    log_y = np.log(df["ETHUSDT"])
    log_x = np.log(df["BTCUSDT"])

    beta_full = _rolling_ols_beta(log_y, log_x, window)

    # Truncate at row 100 and recompute
    beta_truncated = _rolling_ols_beta(log_y.iloc[:101], log_x.iloc[:101], window)

    # Value at position 100 must be identical
    assert abs(float(beta_full.iloc[100]) - float(beta_truncated.iloc[100])) < 1e-10, (
        "β at position 100 differs between full and truncated series — "
        "rolling computation is leaking future data."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Robust z-score: no future data accessed
# ─────────────────────────────────────────────────────────────────────────────

def test_zscore_no_future_data():
    """Z-score at index i must equal z-score computed on data[:i+1]."""
    from src.features.pair_spread import _robust_zscore

    window = 24
    df = _make_price_df(n=200)
    log_y = np.log(df["ETHUSDT"])
    log_x = np.log(df["BTCUSDT"])

    # Use a constant beta for isolation
    spread = log_y - 0.5 * log_x

    z_full = _robust_zscore(spread, window)
    z_truncated = _robust_zscore(spread.iloc[:101], window)

    assert abs(float(z_full.iloc[100]) - float(z_truncated.iloc[100])) < 1e-10, (
        "Robust z-score at position 100 differs — future data leak detected."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. Realized vol: rolling std uses only past data
# ─────────────────────────────────────────────────────────────────────────────

def test_realized_vol_no_future_data():
    """
    Rolling std of log returns must be identical whether computed on
    the full series or on a series truncated at the same point.
    """
    rng = np.random.default_rng(0)
    n = 200
    closes = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 100)
    log_ret = np.log(closes / closes.shift(1))
    window = 24

    vol_full = log_ret.rolling(window).std()
    vol_trunc = log_ret.iloc[:101].rolling(window).std()

    assert abs(float(vol_full.iloc[100]) - float(vol_trunc.iloc[100])) < 1e-10, (
        "Realized vol at position 100 differs — future data leak in rolling std."
    )
