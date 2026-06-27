"""
Phase 3 — Triple-barrier labeling on ou_zscore signal.

Entry condition : |ou_zscore[i]| > TB_ENTRY_Z (default 2.0)
Profit barrier  : |ou_zscore[j]| < TB_EXIT_Z  → tb_label = +1
Stop barrier    : |ou_zscore[j]| > TB_STOP_Z  → tb_label = -1
Vertical barrier: j - i > max_hold_h           → tb_label =  0 (timeout)

Non-overlapping positions enforced via active_until_idx.

Usage:
    python -m src.labels.triple_barrier
    python -m src.labels.triple_barrier --feature-version v0.3-kalman --label-version v1.0-tb
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_VERSION,
    LABEL_VERSION,
    TB_ENTRY_Z,
    TB_EXIT_Z,
    TB_STOP_Z,
    TB_MAX_HOLD_H,
    INTERVAL,
)
from src.db import get_engine, upsert_ml_labels

logger = logging.getLogger(__name__)

PAIR_ID = "ETHUSDT_BTCUSDT"


def _fetch_features(feature_version: str) -> pd.DataFrame:
    """Load ou_zscore + spread_kalman from pair_features, ordered by timestamp."""
    sql = """
        SELECT timestamp, ou_zscore, spread_kalman
        FROM   pair_features
        WHERE  pair_id         = 'ETHUSDT_BTCUSDT'
          AND  interval        = '1h'
          AND  feature_version = %(fv)s
          AND  ou_zscore       IS NOT NULL
          AND  spread_kalman   IS NOT NULL
        ORDER  BY timestamp
    """
    df = pd.read_sql_query(
        sql, get_engine(), params={"fv": feature_version},
        parse_dates=["timestamp"],
    )
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def compute_labels(
    df: pd.DataFrame,
    entry_z: float = TB_ENTRY_Z,
    exit_z: float = TB_EXIT_Z,
    stop_z: float = TB_STOP_Z,
    max_hold_h: int = TB_MAX_HOLD_H,
) -> list[dict]:
    """
    Apply triple-barrier algorithm to ou_zscore series.

    Returns list of label dicts; one dict per non-overlapping entry signal.
    All timestamps are tz-aware UTC.
    """
    timestamps = df["timestamp"].values        # numpy datetime64[ns, UTC]
    ou_zscore = df["ou_zscore"].to_numpy(dtype=float)
    spread_kalman = df["spread_kalman"].to_numpy(dtype=float)
    n = len(ou_zscore)

    rows = []
    active_until_idx = -1  # non-overlapping: skip indices ≤ this after entry

    for i in range(n):
        if i <= active_until_idx:
            continue

        abs_z = abs(ou_zscore[i])
        if abs_z <= entry_z:
            continue

        # Entry signal
        entry_side = -1 if ou_zscore[i] > 0 else 1  # short spread if z>0 (above μ)
        entry_zscore = float(ou_zscore[i])
        entry_spread = float(spread_kalman[i])

        # Scan forward up to max_hold_h bars
        tb_label = 0
        exit_idx = min(i + max_hold_h, n - 1)  # vertical barrier default

        for j in range(i + 1, min(i + max_hold_h + 1, n)):
            abs_zj = abs(ou_zscore[j])
            if abs_zj < exit_z:        # profit barrier hit
                tb_label = 1
                exit_idx = j
                break
            if abs_zj > stop_z:        # stop barrier hit
                tb_label = -1
                exit_idx = j
                break
        # else: vertical barrier, tb_label stays 0

        exit_zscore = float(ou_zscore[exit_idx])
        exit_spread = float(spread_kalman[exit_idx])

        delta_zscore = abs(entry_zscore) - abs(exit_zscore)
        # Raw log-return of the spread position (no division by spread_kalman).
        # Dividing by spread_kalman would explode: Kalman β minimizes spread variance,
        # so spread_kalman ≈ 0, making any ratio meaningless.
        # As a log-return: ≈ % P&L of the market-neutral portfolio for small moves.
        spread_return = entry_side * (exit_spread - entry_spread)

        rows.append({
            "pair_id": PAIR_ID,
            "interval": INTERVAL,
            "entry_timestamp": pd.Timestamp(timestamps[i]).to_pydatetime(),
            "exit_timestamp": pd.Timestamp(timestamps[exit_idx]).to_pydatetime(),
            "entry_side": entry_side,
            "entry_zscore": entry_zscore,
            "hold_bars": exit_idx - i,
            "tb_label": tb_label,
            "delta_zscore": delta_zscore,
            "spread_return": spread_return,
            "stop_z": stop_z,
            "max_hold_h": max_hold_h,
        })

        active_until_idx = exit_idx

    return rows


def compute_and_store(
    feature_version: str,
    label_version: str,
    entry_z: float = TB_ENTRY_Z,
    exit_z: float = TB_EXIT_Z,
    stop_z: float = TB_STOP_Z,
    max_hold_h: int = TB_MAX_HOLD_H,
) -> int:
    """
    Load pair_features, apply triple-barrier labeling, upsert into ml_labels.
    Returns number of rows inserted.
    """
    df = _fetch_features(feature_version)
    if df.empty:
        logger.warning("No pair_features rows for version=%s — skipping", feature_version)
        return 0

    logger.info(
        "Running triple-barrier labeling (N=%d, entry_z=%.1f, exit_z=%.1f, "
        "stop_z=%.1f, max_hold=%dh)",
        len(df), entry_z, exit_z, stop_z, max_hold_h,
    )

    label_rows = compute_labels(df, entry_z=entry_z, exit_z=exit_z,
                                stop_z=stop_z, max_hold_h=max_hold_h)

    for r in label_rows:
        r["feature_version"] = feature_version
        r["label_version"] = label_version

    if not label_rows:
        logger.warning("No entry signals found — check TB_ENTRY_Z threshold")
        return 0

    n = upsert_ml_labels(label_rows)
    pos = sum(1 for r in label_rows if r["tb_label"] == 1)
    neg = sum(1 for r in label_rows if r["tb_label"] == -1)
    timeout = sum(1 for r in label_rows if r["tb_label"] == 0)
    logger.info(
        "Inserted %d ml_labels (label_version=%s): +1=%d, -1=%d, 0=%d",
        n, label_version, pos, neg, timeout,
    )
    return n


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Triple-barrier labeling on ou_zscore")
    parser.add_argument("--feature-version", default=FEATURE_VERSION, dest="feature_version")
    parser.add_argument("--label-version", default=LABEL_VERSION, dest="label_version")
    parser.add_argument("--entry-z", type=float, default=TB_ENTRY_Z, dest="entry_z")
    parser.add_argument("--exit-z", type=float, default=TB_EXIT_Z, dest="exit_z")
    parser.add_argument("--stop-z", type=float, default=TB_STOP_Z, dest="stop_z")
    parser.add_argument("--max-hold-h", type=int, default=TB_MAX_HOLD_H, dest="max_hold_h")
    args = parser.parse_args()

    compute_and_store(
        args.feature_version, args.label_version,
        entry_z=args.entry_z, exit_z=args.exit_z,
        stop_z=args.stop_z, max_hold_h=args.max_hold_h,
    )


if __name__ == "__main__":
    main()
