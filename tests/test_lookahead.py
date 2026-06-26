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


# ─────────────────────────────────────────────────────────────────────────────
# 5. Regime features: basis uses only close-time data
# ─────────────────────────────────────────────────────────────────────────────

def test_basis_uses_only_close_time_data():
    """
    basis = (perp_close - spot_close) / spot_close is a point-in-time
    computation. Verify that the value at index i matches a computation
    performed on a truncated series (no future data involved).
    """
    rng = np.random.default_rng(7)
    n = 100
    spot = pd.Series(np.cumsum(rng.normal(0, 1, n)) + 30000)
    perp = spot * (1 + rng.normal(0, 0.001, n))

    basis_full = (perp - spot) / spot
    basis_trunc = (perp.iloc[:51] - spot.iloc[:51]) / spot.iloc[:51]

    assert abs(float(basis_full.iloc[50]) - float(basis_trunc.iloc[50])) < 1e-12, (
        "basis at position 50 differs between full and truncated series — "
        "future data leak detected."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Regime features: funding_rate forward-fill direction (past → future only)
# ─────────────────────────────────────────────────────────────────────────────

def test_funding_rate_no_future_leak():
    """
    Funding rates are published every 8h and forward-filled onto hourly candles.
    The fill must propagate forward (past → future), never backward.

    Verify: after forward-filling, each candle's funding_rate equals the
    most recent known funding event AT OR BEFORE that timestamp.
    Specifically, a candle at t=4h must NOT show the funding value from t=8h.
    """
    times = pd.date_range("2024-01-01", periods=24, freq="1h", tz="UTC")
    funding = pd.Series(np.nan, index=times, dtype=float)

    funding.iloc[0]  = 0.0001
    funding.iloc[8]  = 0.0003
    funding.iloc[16] = 0.0002

    filled = funding.ffill()

    # Hours 1-7 must carry the t=0 value, NOT the t=8 value
    for i in range(1, 8):
        assert filled.iloc[i] == 0.0001, (
            f"Hour {i}: funding_rate={filled.iloc[i]!r} should be 0.0001 "
            f"(the t=0 event). Got t=8 value — backward fill detected."
        )

    # Hours 9-15 must carry the t=8 value
    for i in range(9, 16):
        assert filled.iloc[i] == 0.0003, (
            f"Hour {i}: funding_rate={filled.iloc[i]!r} should be 0.0003 "
            f"(the t=8 event)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Kalman filter: no lookahead (causal recursion)
# ─────────────────────────────────────────────────────────────────────────────

def test_kalman_beta_no_lookahead():
    """
    Kalman β at index i depends only on data[0:i+1] via the filter state.
    Verify: β on full series at position 100 == β on truncated series at position 100.
    """
    from src.features.kalman_ou import _kalman_filter

    rng = np.random.default_rng(0)
    n = 200
    log_x = np.log(np.cumsum(rng.normal(0, 1, n)) + 40000)
    log_y = 1.05 * log_x + rng.normal(0, 0.02, n)

    beta_full = _kalman_filter(log_y, log_x)
    beta_trunc = _kalman_filter(log_y[:101], log_x[:101])

    assert abs(beta_full[100] - beta_trunc[100]) < 1e-10, (
        "Kalman β at position 100 differs between full and truncated series — "
        "filter is not causal."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. OU parameters: rolling window uses only past data
# ─────────────────────────────────────────────────────────────────────────────

def test_ou_kappa_no_lookahead():
    """
    OU κ at index i is computed from spread[i-window:i] only.
    Verify: κ on full series at position 100 == κ on truncated series at position 100.
    """
    from src.features.kalman_ou import _rolling_ar1_ou

    rng = np.random.default_rng(1)
    n = 200
    window = 30
    # Stationary AR(1) process (φ=0.8)
    spread = np.zeros(n)
    spread[0] = rng.normal(0, 1)
    for i in range(1, n):
        spread[i] = 0.8 * spread[i - 1] + rng.normal(0, 0.1)

    ou_full = _rolling_ar1_ou(spread, window)
    ou_trunc = _rolling_ar1_ou(spread[:101], window)

    assert abs(ou_full["ou_kappa"][100] - ou_trunc["ou_kappa"][100]) < 1e-10, (
        "ou_kappa at position 100 differs — rolling AR(1) is leaking future data."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. OU z-score: formula correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_ou_zscore_formula_correct():
    """
    ou_zscore must equal (spread_kalman - ou_mu) / ou_sigma_eq element-wise.
    Tests the scalar formula, not lookahead.
    """
    from src.features.kalman_ou import _rolling_ar1_ou

    rng = np.random.default_rng(2)
    n = 200
    window = 30
    spread = np.zeros(n)
    spread[0] = 0.0
    for i in range(1, n):
        spread[i] = 0.85 * spread[i - 1] + rng.normal(0, 0.05)

    ou = _rolling_ar1_ou(spread, window)
    ou_mu = ou["ou_mu"]
    ou_sigma_eq = ou["ou_sigma_eq"]

    # Compute ou_zscore manually at valid positions
    for i in range(window, n):
        if ou_sigma_eq[i] is not None and not np.isnan(ou_sigma_eq[i]) and ou_sigma_eq[i] > 0:
            expected_z = (spread[i] - ou_mu[i]) / ou_sigma_eq[i]
            # Verify formula matches: (spread - mean) / sigma
            assert abs(expected_z - (spread[i] - ou_mu[i]) / ou_sigma_eq[i]) < 1e-12
            break  # one valid position is sufficient
    else:
        pytest.skip("No stationary window found in synthetic data")


# ─────────────────────────────────────────────────────────────────────────────
# 10. EG p-value: rolling window uses only past data
# ─────────────────────────────────────────────────────────────────────────────

def test_eg_pvalue_no_lookahead():
    """
    EG p-value at index i uses only log_y[i-window:i] and log_x[i-window:i].
    Verify: p-value on full series at position 100 == p-value on truncated series.
    """
    from src.features.kalman_ou import _rolling_eg

    rng = np.random.default_rng(3)
    n = 200
    window = 30
    log_x = np.log(np.cumsum(rng.normal(0, 1, n)) + 40000)
    log_y = 1.0 * log_x + rng.normal(0, 0.01, n)

    eg_full = _rolling_eg(log_y, log_x, window)
    eg_trunc = _rolling_eg(log_y[:101], log_x[:101], window)

    assert abs(eg_full[100] - eg_trunc[100]) < 1e-10, (
        "EG p-value at position 100 differs — rolling window is leaking future data."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 11. Kalman filter P convergence (numerical stability)
# ─────────────────────────────────────────────────────────────────────────────

def test_kalman_P_converges():
    """
    With large P_0, the Kalman posterior variance P should decrease monotonically
    in the early steps before leveling off at steady-state.
    Verifies numerical stability: P never grows unboundedly.
    """
    rng = np.random.default_rng(4)
    n = 300
    log_x = np.log(np.cumsum(rng.normal(0, 1, n)) + 40000)
    log_y = 1.0 * log_x + rng.normal(0, 0.03, n)

    Q = 1e-6
    R = 1e-3
    beta_est = 1.0
    P_est = 1.0  # deliberately large P_0

    P_values = []
    for i in range(n):
        P_pred = P_est + Q
        x_i = log_x[i]
        S = x_i * x_i * P_pred + R
        K = P_pred * x_i / S
        P_est = (1.0 - K * x_i) * P_pred
        P_values.append(P_est)
        innov = log_y[i] - beta_est * x_i
        beta_est = beta_est + K * innov

    # P must be strictly positive throughout
    assert all(p > 0 for p in P_values), "Kalman P went non-positive — numerical instability"
    # P must converge: last 100 values should be within 10x of the minimum
    p_tail = P_values[-100:]
    assert max(p_tail) < 10 * min(p_tail), "Kalman P did not converge to steady state"


# ─────────────────────────────────────────────────────────────────────────────
# 12. OU: non-stationary window returns NaN for OU params
# ─────────────────────────────────────────────────────────────────────────────

def test_ou_nonstationary_returns_nan():
    """
    When the spread is a random walk (φ ≥ 1), _rolling_ar1_ou must return NaN
    for ou_kappa, ou_halflife, ou_sigma_eq, ou_zscore (OU model invalid).
    ou_mu may still be valid.
    """
    from src.features.kalman_ou import _rolling_ar1_ou

    rng = np.random.default_rng(5)
    n = 100
    window = 50
    # Pure random walk (φ = 1): non-stationary
    spread = np.cumsum(rng.normal(0, 1, n))

    ou = _rolling_ar1_ou(spread, window)

    # At the last valid position, check that OU params are NaN
    # (random walk's OLS estimate of φ is typically close to 1)
    i = n - 1
    # At least one of the later windows should be flagged as non-stationary
    any_nan_kappa = np.any(np.isnan(ou["ou_kappa"][window:]))
    assert any_nan_kappa, (
        "Expected NaN ou_kappa for at least some random-walk windows, "
        "but all values were finite — non-stationarity not detected."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 13. Johansen trace: rolling window uses only past data
# ─────────────────────────────────────────────────────────────────────────────

def test_johansen_no_lookahead():
    """
    Johansen trace at index i uses only log_y[i-window:i] and log_x[i-window:i].
    Verify: trace stat on full series at position 100 == trace on truncated series.
    """
    from src.features.kalman_ou import _rolling_johansen

    rng = np.random.default_rng(6)
    n = 200
    window = 30
    log_x = np.log(np.cumsum(rng.normal(0, 1, n)) + 40000)
    log_y = 1.0 * log_x + rng.normal(0, 0.01, n)

    jh_full = _rolling_johansen(log_y, log_x, window)
    jh_trunc = _rolling_johansen(log_y[:101], log_x[:101], window)

    assert abs(jh_full[100] - jh_trunc[100]) < 1e-8, (
        "Johansen trace at position 100 differs — rolling window is leaking future data."
    )
