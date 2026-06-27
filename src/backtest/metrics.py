"""
Phase 4 — Backtest performance metrics.
"""
import numpy as np


def sharpe_ratio(returns: np.ndarray, trades_per_year: float) -> float:
    """
    Annualized Sharpe ratio on a per-trade return series.

    Annualizes by scaling with sqrt(trades_per_year) — each 'period' is one trade.
    Returns NaN if std is zero or fewer than 2 observations.
    """
    if len(returns) < 2:
        return float("nan")
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return float("nan")
    return float(np.mean(returns) / std * np.sqrt(trades_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown as a positive fraction.

    equity_curve: cumulative product array starting from 1.0
    Returns 0.0 if equity never falls below a previous peak.
    """
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak
    return float(np.max(drawdowns))


def calmar_ratio(annualized_return: float, max_dd: float) -> float:
    """Calmar = annualized return / max drawdown. NaN if max_dd is zero."""
    if max_dd == 0:
        return float("nan")
    return annualized_return / max_dd
