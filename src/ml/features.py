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
    entry_ts = pd.to_datetime(df["entry_timestamp"], utc=True)

    ou_zscore = df["ou_zscore"].astype(float)
    eg_pvalue = df["eg_pvalue"].astype(float)
    johansen_trace = df["johansen_trace"].astype(float)
    basis_y = df["basis_y"].astype(float)
    basis_x = df["basis_x"].astype(float)
    taker_ratio_y = df["taker_ratio_y"].astype(float)
    taker_ratio_x = df["taker_ratio_x"].astype(float)

    X = pd.DataFrame({
        # Tier 1
        "ou_zscore": ou_zscore,
        "abs_ou_zscore": ou_zscore.abs(),
        "ou_halflife": df["ou_halflife"].astype(float),
        "ou_kappa": df["ou_kappa"].astype(float),
        "ou_sigma_eq": df["ou_sigma_eq"].astype(float),
        # Tier 2
        "eg_pvalue": eg_pvalue,
        "johansen_trace": johansen_trace,
        "is_cointegrated": (eg_pvalue < 0.05).astype(int),
        "johansen_sig": (johansen_trace > _JOHANSEN_95).astype(int),
        # Tier 3
        "funding_spread": df["funding_spread"].astype(float),
        "basis_y": basis_y,
        "basis_x": basis_x,
        "basis_diff": basis_y - basis_x,
        "taker_ratio_y": taker_ratio_y,
        "taker_ratio_x": taker_ratio_x,
        "taker_diff": taker_ratio_y - taker_ratio_x,
        "spread_std": df["spread_std"].astype(float),
        # Tier 4 — derived from entry_timestamp (tz-aware UTC)
        "hour_of_day": entry_ts.dt.hour.astype(float).values,
        "day_of_week": entry_ts.dt.dayofweek.astype(float).values,
    }, index=df.index)

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
