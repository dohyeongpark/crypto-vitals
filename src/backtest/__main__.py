"""
Phase 4 — Backtest CLI.

Loads ml_labels with meta_label, runs P&L simulation, prints results table.

Usage:
    python -m src.backtest
    python -m src.backtest --fee 0.004 --slippage 0.002 --label-version v1.0-tb
"""
import argparse
import logging
import time

from src.config import FEATURE_VERSION, LABEL_VERSION
from src.db import load_labels_for_training, get_engine
from src.backtest.engine import run_backtest, FEE_PER_RT, SLIPPAGE_PER_RT

import pandas as pd


def _load_with_meta(feature_version: str, label_version: str) -> pd.DataFrame:
    """load_labels_for_training + meta_label column from ml_labels."""
    base = load_labels_for_training(feature_version, label_version)
    if base.empty:
        return base

    meta_sql = """
        SELECT entry_timestamp, meta_label, meta_prob
        FROM ml_labels
        WHERE feature_version = %(fv)s AND label_version = %(lv)s
          AND meta_prob IS NOT NULL
    """
    meta = pd.read_sql_query(
        meta_sql, get_engine(),
        params={"fv": feature_version, "lv": label_version},
        parse_dates=["entry_timestamp"],
    )
    meta["entry_timestamp"] = pd.to_datetime(meta["entry_timestamp"], utc=True)
    base["entry_timestamp"] = pd.to_datetime(base["entry_timestamp"], utc=True)
    return base.merge(meta, on="entry_timestamp", how="inner")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Phase 4 backtest — meta-labeled stat-arb P&L")
    parser.add_argument("--feature-version", default=FEATURE_VERSION, dest="feature_version")
    parser.add_argument("--label-version", default=LABEL_VERSION, dest="label_version")
    parser.add_argument("--fee", type=float, default=FEE_PER_RT,
                        help="Round-trip fee fraction (default 0.004 = 0.4%%)")
    parser.add_argument("--slippage", type=float, default=SLIPPAGE_PER_RT,
                        help="Round-trip slippage fraction (default 0.002 = 0.2%%)")
    args = parser.parse_args()

    logger.info("Loading ml_labels (fv=%s, lv=%s)", args.feature_version, args.label_version)
    df = _load_with_meta(args.feature_version, args.label_version)
    if df.empty:
        logger.error("No data found — run Step 8 (triple-barrier) and src.ml.predict first")
        return

    total = len(df)
    selected = int((df["meta_label"] == 1).sum())
    logger.info("Total signals: %d | meta_label=1: %d | meta_label=0: %d",
                total, selected, total - selected)

    # Backtest: meta_label=1 only
    res = run_backtest(df, fee_per_rt=args.fee, slippage_per_rt=args.slippage)

    total_cost_pct = (args.fee + args.slippage) * 100

    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS  |  {args.label_version}  |  meta_label=1 only")
    print("=" * 60)
    print(f"  Trades selected      : {res['n_trades']}")
    print(f"  Years of data        : {res.get('years_of_data', 0):.2f}")
    print(f"  Trades per year      : {res.get('trades_per_year', 0):.0f}")
    print(f"  Win rate             : {res['win_rate']:.1%}")
    print(f"  Cost per RT          : {total_cost_pct:.2f}%  (fee={args.fee*100:.2f}% + slip={args.slippage*100:.2f}%)")
    print("-" * 60)
    print(f"  Avg gross return     : {res['avg_gross_return']*100:+.3f}%")
    print(f"  Avg net return       : {res['avg_net_return']*100:+.3f}%")
    print(f"  Total gross          : {res['total_gross_pct']:+.2f}%")
    print(f"  Total net            : {res['total_net_pct']:+.2f}%")
    print(f"  Annualized return    : {res.get('annualized_return_pct', float('nan')):+.2f}%")
    print("-" * 60)
    print(f"  Sharpe (annual)      : {res['sharpe_annual']:.3f}")
    print(f"  Max drawdown         : {res['max_drawdown_pct']:.2f}%")
    print(f"  Calmar ratio         : {res['calmar']:.3f}")
    print("=" * 60 + "\n")

    # Also show unfiltered baseline for comparison
    df_all = df.copy()
    df_all["meta_label"] = 1  # pretend all selected
    res_all = run_backtest(df_all, fee_per_rt=args.fee, slippage_per_rt=args.slippage)
    print("  BASELINE (all signals, no meta filter):")
    print(f"    n={res_all['n_trades']}  win={res_all['win_rate']:.1%}"
          f"  avg_net={res_all['avg_net_return']*100:+.3f}%"
          f"  Sharpe={res_all['sharpe_annual']:.3f}"
          f"  MaxDD={res_all['max_drawdown_pct']:.2f}%")
    print()


if __name__ == "__main__":
    main()
