"""
Phase 3 — Feature matrix builder for meta-labeling.

All features are derived from pair_features at entry_timestamp (causal).
No lookahead: the label is in ml_labels, not in this feature set.
"""
import numpy as np
import pandas as pd

# Johansen 95% critical value for 2 variables (r=0)
_JOHANSEN_95 = 15.41

FEATURE_NAMES = [
    # Tier 1 — OU signal
    "ou_zscore",
    "abs_ou_zscore",
    "ou_halflife",
    "ou_kappa",
    "ou_sigma_eq",
    # Tier 2 — cointegration quality
    "eg_pvalue",
    "johansen_trace",
    "is_cointegrated",    # int(eg_pvalue < 0.05)
    "johansen_sig",       # int(johansen_trace > 15.41)
    # Tier 3 — regime / microstructure
    "funding_spread",
    "basis_y",
    "basis_x",
    "basis_diff",         # basis_y - basis_x
    "taker_ratio_y",
    "taker_ratio_x",
    "taker_diff",         # taker_ratio_y - taker_ratio_x
    "spread_std",
    # Tier 4 — calendar
    "hour_of_day",
    "day_of_week",
]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build feature matrix from a DataFrame returned by load_labels_for_training().

    Returns (X, feature_names) where X has one row per ml_labels entry and
    columns matching FEATURE_NAMES. Rows with any NaN are dropped.
    """
    X = pd.DataFrame(index=df.index)

    # Tier 1
    X["ou_zscore"] = df["ou_zscore"].astype(float)
    X["abs_ou_zscore"] = X["ou_zscore"].abs()
    X["ou_halflife"] = df["ou_halflife"].astype(float)
    X["ou_kappa"] = df["ou_kappa"].astype(float)
    X["ou_sigma_eq"] = df["ou_sigma_eq"].astype(float)

    # Tier 2
    X["eg_pvalue"] = df["eg_pvalue"].astype(float)
    X["johansen_trace"] = df["johansen_trace"].astype(float)
    X["is_cointegrated"] = (X["eg_pvalue"] < 0.05).astype(int)
    X["johansen_sig"] = (X["johansen_trace"] > _JOHANSEN_95).astype(int)

    # Tier 3
    X["funding_spread"] = df["funding_spread"].astype(float)
    X["basis_y"] = df["basis_y"].astype(float)
    X["basis_x"] = df["basis_x"].astype(float)
    X["basis_diff"] = X["basis_y"] - X["basis_x"]
    X["taker_ratio_y"] = df["taker_ratio_y"].astype(float)
    X["taker_ratio_x"] = df["taker_ratio_x"].astype(float)
    X["taker_diff"] = X["taker_ratio_y"] - X["taker_ratio_x"]
    X["spread_std"] = df["spread_std"].astype(float)

    # Tier 4 — derived from entry_timestamp (tz-aware UTC)
    entry_ts = pd.to_datetime(df["entry_timestamp"], utc=True)
    X["hour_of_day"] = entry_ts.dt.hour.astype(float)
    X["day_of_week"] = entry_ts.dt.dayofweek.astype(float)

    # Keep only declared feature order
    X = X[FEATURE_NAMES]

    n_before = len(X)
    X = X.dropna()
    if len(X) < n_before:
        import logging
        logging.getLogger(__name__).warning(
            "Dropped %d rows with NaN features (%d → %d)",
            n_before - len(X), n_before, len(X),
        )

    return X, FEATURE_NAMES
