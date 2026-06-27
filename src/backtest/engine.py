"""
Phase 4 — P&L simulation engine for meta-labeled stat-arb signals.

Cost model (per round trip):
  fee_per_rt      = 0.004  (0.1% taker × 4 legs: ETH+BTC entry + ETH+BTC exit)
  slippage_per_rt = 0.002  (0.05% market impact × 4 legs)
  total cost      = 0.006 (0.6%) per round trip

spread_return is pre-computed in ml_labels as:
  entry_side × (spread_kalman_exit − spread_kalman_entry) / |spread_kalman_entry|
"""
import numpy as np
import pandas as pd

from src.backtest.metrics import sharpe_ratio, max_drawdown, calmar_ratio

# Default cost assumptions (Binance spot, taker fee 0.1%)
FEE_PER_RT = 0.004
SLIPPAGE_PER_RT = 0.002


def run_backtest(
    df: pd.DataFrame,
    fee_per_rt: float = FEE_PER_RT,
    slippage_per_rt: float = SLIPPAGE_PER_RT,
    meta_label_col: str = "meta_label",
) -> dict:
    """
    Simulate P&L on rows where meta_label == 1.

    Parameters
    ----------
    df : DataFrame from load_labels_for_training() with meta_label column added.
         Must contain: entry_timestamp, exit_timestamp, spread_return, tb_label, meta_label.
    fee_per_rt       : round-trip fee fraction
    slippage_per_rt  : round-trip slippage fraction
    meta_label_col   : column name for meta-label filter

    Returns
    -------
    dict with scalar metrics. Returns all-NaN dict if no trades selected.
    """
    total_cost = fee_per_rt + slippage_per_rt

    selected = df[df[meta_label_col] == 1].copy()
    selected = selected.sort_values("entry_timestamp").reset_index(drop=True)

    n_trades = len(selected)
    if n_trades == 0:
        return {
            "n_trades": 0, "win_rate": float("nan"),
            "avg_gross_return": float("nan"), "avg_net_return": float("nan"),
            "sharpe_annual": float("nan"), "max_drawdown_pct": float("nan"),
            "calmar": float("nan"), "total_gross_pct": float("nan"),
            "total_net_pct": float("nan"), "years_of_data": float("nan"),
        }

    gross = selected["spread_return"].astype(float).to_numpy()
    net = gross - total_cost

    # Win rate: tb_label == +1 among selected trades
    win_rate = float((selected["tb_label"] == 1).mean())

    # Equity curve (compounded)
    equity = np.cumprod(1.0 + net)

    # Time span for annualization
    t_min = pd.to_datetime(selected["entry_timestamp"], utc=True).min()
    t_max = pd.to_datetime(selected["exit_timestamp"], utc=True).max()
    years = max((t_max - t_min).total_seconds() / (365.25 * 24 * 3600), 1 / 365)
    trades_per_year = n_trades / years

    ann_return = float(equity[-1] ** (1 / years) - 1)
    sharpe = sharpe_ratio(net, trades_per_year)
    max_dd = max_drawdown(equity)
    calmar = calmar_ratio(ann_return, max_dd)

    return {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_gross_return": float(np.mean(gross)),
        "avg_net_return": float(np.mean(net)),
        "sharpe_annual": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "calmar": calmar,
        "total_gross_pct": float((equity[-1] - 1) * 100 + total_cost * n_trades * 100),
        "total_net_pct": float((equity[-1] - 1) * 100),
        "years_of_data": years,
        "trades_per_year": trades_per_year,
        "annualized_return_pct": ann_return * 100,
    }
