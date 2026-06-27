"""
Phase 3 — Purged walk-forward cross-validation (Lopez de Prado, 2018).

Key idea: in time-series ML with overlapping labels (triple-barrier labels span
multiple bars), naive k-fold leaks information because training samples near the
test boundary have exit_timestamps that fall inside the test period.

Fix:
  - Purge: remove training samples whose exit_timestamp >= test_start
  - Embargo: also remove training samples within `embargo_h` bars before test_start
    (in case of near-boundary serial correlation)

Returns expanding-window folds (increasing train size), skipping fold 0
(no train data before the first test window).
"""
from datetime import timedelta

import numpy as np
import pandas as pd


def purged_walk_forward_cv(
    entry_timestamps: pd.Series,
    exit_timestamps: pd.Series,
    n_folds: int = 5,
    embargo_h: int = 12,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, test_idx) pairs for purged walk-forward CV.

    Parameters
    ----------
    entry_timestamps : pd.Series of tz-aware datetimes, length N
    exit_timestamps  : pd.Series of tz-aware datetimes, length N
    n_folds          : number of time splits (test windows)
    embargo_h        : hours before test_start to exclude from training

    Returns
    -------
    List of (train_indices, test_indices) as numpy arrays.
    Fold 0 (no training data) is omitted → typically n_folds-1 usable folds.
    """
    entry_ts = pd.to_datetime(entry_timestamps, utc=True).reset_index(drop=True)
    exit_ts = pd.to_datetime(exit_timestamps, utc=True).reset_index(drop=True)
    n = len(entry_ts)

    t_min = entry_ts.min()
    t_max = entry_ts.max()
    total_span = t_max - t_min
    fold_width = total_span / n_folds

    folds = []
    for k in range(1, n_folds):  # skip k=0 (no train data)
        test_start = t_min + k * fold_width
        test_end = t_min + (k + 1) * fold_width

        # Test: entries in [test_start, test_end)
        test_mask = (entry_ts >= test_start) & (entry_ts < test_end)
        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0:
            continue

        embargo_cutoff = test_start - timedelta(hours=embargo_h)

        # Train: entries before test_start, excluding:
        #   1. Purge: exit_timestamp >= test_start (label overlaps with test)
        #   2. Embargo: entry_timestamp >= embargo_cutoff
        train_mask = (
            (entry_ts < test_start)
            & (exit_ts < test_start)           # purge
            & (entry_ts < embargo_cutoff)      # embargo
        )
        train_idx = np.where(train_mask)[0]
        if len(train_idx) == 0:
            continue

        folds.append((train_idx, test_idx))

    return folds
