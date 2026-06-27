"""
Phase 4 backtest formula tests.
Verify metrics.py arithmetic without requiring a live DB.
"""
import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sharpe ratio formula
# ─────────────────────────────────────────────────────────────────────────────

def test_sharpe_formula():
    """
    Sharpe = mean(returns) / std(returns) * sqrt(trades_per_year).
    Verify on a known return series.
    """
    from src.backtest.metrics import sharpe_ratio

    returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01, -0.003, 0.012, 0.008])
    trades_per_year = 100.0

    expected = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(trades_per_year)
    computed = sharpe_ratio(returns, trades_per_year)

    assert abs(computed - expected) < 1e-10, (
        f"Sharpe formula mismatch: expected {expected:.6f}, got {computed:.6f}"
    )


def test_sharpe_single_observation_returns_nan():
    """With < 2 observations, Sharpe must return NaN (std undefined)."""
    from src.backtest.metrics import sharpe_ratio

    assert np.isnan(sharpe_ratio(np.array([0.01]), 100.0)), (
        "Sharpe on single observation should be NaN"
    )


def test_sharpe_zero_std_returns_nan():
    """Constant return series → std=0 → Sharpe must return NaN (not inf)."""
    from src.backtest.metrics import sharpe_ratio

    returns = np.full(10, 0.01)
    assert np.isnan(sharpe_ratio(returns, 100.0)), (
        "Sharpe with zero std should be NaN, not inf"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Max drawdown formula
# ─────────────────────────────────────────────────────────────────────────────

def test_max_drawdown_formula():
    """
    equity_curve = [1.0, 1.1, 0.9, 0.95, 0.75]
    Peak sequence : [1.0, 1.1, 1.1, 1.1, 1.1]
    DD sequence   : [0, 0, (1.1-0.9)/1.1, (1.1-0.95)/1.1, (1.1-0.75)/1.1]
    Max DD        : (1.1-0.75)/1.1 = 0.35/1.1 ≈ 0.31818...
    """
    from src.backtest.metrics import max_drawdown

    equity = np.array([1.0, 1.1, 0.9, 0.95, 0.75])
    expected = (1.1 - 0.75) / 1.1  # ≈ 0.31818

    computed = max_drawdown(equity)
    assert abs(computed - expected) < 1e-10, (
        f"MaxDD formula mismatch: expected {expected:.6f}, got {computed:.6f}"
    )


def test_max_drawdown_monotone_increase():
    """Monotonically increasing equity → drawdown = 0."""
    from src.backtest.metrics import max_drawdown

    equity = np.array([1.0, 1.05, 1.12, 1.20, 1.35])
    assert max_drawdown(equity) == 0.0, (
        "Monotone increasing equity should have MaxDD = 0"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fee deduction in run_backtest
# ─────────────────────────────────────────────────────────────────────────────

def test_fee_deducted():
    """
    net_return = spread_return - fee_per_rt - slippage_per_rt exactly.
    Verify avg_net_return = avg_gross_return - total_cost.
    """
    import pandas as pd
    from src.backtest.engine import run_backtest

    fee = 0.004
    slip = 0.002
    total_cost = fee + slip

    gross_returns = [0.01, 0.02, -0.005, 0.015]
    df = pd.DataFrame({
        "entry_timestamp": pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"),
        "exit_timestamp": pd.date_range("2024-01-01 04:00", periods=4, freq="1h", tz="UTC"),
        "spread_return": gross_returns,
        "tb_label": [1, 1, -1, 1],
        "meta_label": [1, 1, 1, 1],
    })

    res = run_backtest(df, fee_per_rt=fee, slippage_per_rt=slip)

    expected_avg_gross = np.mean(gross_returns)
    expected_avg_net = expected_avg_gross - total_cost

    assert abs(res["avg_gross_return"] - expected_avg_gross) < 1e-10, (
        f"avg_gross_return mismatch: {res['avg_gross_return']:.8f} vs {expected_avg_gross:.8f}"
    )
    assert abs(res["avg_net_return"] - expected_avg_net) < 1e-10, (
        f"avg_net_return should be avg_gross - {total_cost}: "
        f"{res['avg_net_return']:.8f} vs {expected_avg_net:.8f}"
    )
