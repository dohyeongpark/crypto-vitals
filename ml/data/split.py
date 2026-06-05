"""
Chronological train / val / test split on valid window indices.

Splits the index array (not the data) so each subset can be passed to
VolatilityDataset independently.  Time ordering is always preserved.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)


def time_split(
    indices:    np.ndarray,
    ts:         np.ndarray,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Split index and timestamp arrays chronologically.

    Args:
        indices:    Valid window-endpoint indices from get_valid_indices().
        ts:         Corresponding timestamps (same length as indices).
        val_ratio:  Fraction for validation set  (default 0.15).
        test_ratio: Fraction for test set        (default 0.15).

    Returns:
        {"train": (indices, ts), "val": (indices, ts), "test": (indices, ts)}
    """
    n       = len(indices)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": (indices[:n_train],                        ts[:n_train]),
        "val":   (indices[n_train : n_train + n_val],       ts[n_train : n_train + n_val]),
        "test":  (indices[n_train + n_val :],               ts[n_train + n_val :]),
    }

    for name, (idx, tss) in splits.items():
        log.info(
            "%-5s  %6d samples  [%s → %s]",
            name, len(idx),
            str(tss[0])[:10],
            str(tss[-1])[:10],
        )

    return splits
