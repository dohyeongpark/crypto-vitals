"""
Phase 4 — Hyperparameter grid search for TB + LightGBM parameters.

Grid:
  entry_z     : [1.5, 2.0, 2.5]
  stop_z      : [2.5, 3.0, 3.5]  (only stop_z > entry_z combos)
  n_estimators: [200, 300, 500]
  max_depth   : [3, 4, 5]

= 8 valid TB pairs × 9 LightGBM combos = 72 total combinations.

Objective: mean CV net return on meta_prob > 0.5 predictions across purged folds.
This directly estimates expected trade profitability (vs AUC which is classification quality).

pair_features data is loaded once and reused across all combos to minimize DB round trips.

Usage:
    python -m src.ml.hparam
    python -m src.ml.hparam --feature-version v0.3-kalman --label-version v1.0-tb
"""
import argparse
import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_VERSION, LABEL_VERSION, INTERVAL
from src.db import get_engine
from src.labels.triple_barrier import compute_labels, PAIR_ID
from src.ml.features import build_feature_matrix
from src.ml.cv import purged_walk_forward_cv
from src.backtest.engine import FEE_PER_RT

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

# Grid definition
TB_ENTRY_Z_GRID   = [1.5, 2.0, 2.5]
TB_STOP_Z_GRID    = [2.5, 3.0, 3.5]
N_ESTIMATORS_GRID = [200, 300, 500]
MAX_DEPTH_GRID    = [3, 4, 5]

_BASE_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 15,        # updated per max_depth in loop
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
}


def _fetch_pair_features(feature_version: str) -> pd.DataFrame:
    """Load all pair_features columns needed for TB + feature matrix in one query."""
    sql = """
        SELECT
            timestamp, ou_zscore, spread_kalman,
            ou_halflife, ou_kappa, ou_sigma_eq, ou_mu,
            eg_pvalue, johansen_trace,
            basis_y, basis_x, taker_ratio_y, taker_ratio_x,
            funding_spread, spread_std
        FROM pair_features
        WHERE pair_id         = 'ETHUSDT_BTCUSDT'
          AND interval        = '1h'
          AND feature_version = %(fv)s
          AND ou_zscore       IS NOT NULL
          AND spread_kalman   IS NOT NULL
        ORDER BY timestamp
    """
    df = pd.read_sql_query(
        sql, get_engine(), params={"fv": feature_version},
        parse_dates=["timestamp"],
    )
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    return df


def _build_training_df(label_rows: list[dict], df_pf: pd.DataFrame) -> pd.DataFrame:
    """In-memory join of TB label rows with pair_features on entry_timestamp."""
    if not label_rows:
        return pd.DataFrame()

    labels_df = pd.DataFrame(label_rows)
    labels_df["entry_timestamp"] = pd.to_datetime(labels_df["entry_timestamp"], utc=True)
    labels_df["exit_timestamp"] = pd.to_datetime(labels_df["exit_timestamp"], utc=True)

    pf = df_pf.rename(columns={"timestamp": "entry_timestamp"})
    merged = labels_df.merge(pf, on="entry_timestamp", how="inner")
    return merged


def _cv_net_return(
    X: pd.DataFrame,
    y: pd.Series,
    spread_returns: np.ndarray,
    entry_ts: pd.Series,
    exit_ts: pd.Series,
    lgbm_params: dict,
    threshold: float = 0.5,
    fee_per_rt: float = FEE_PER_RT,
) -> float:
    """
    Run purged walk-forward CV and return mean net return on meta_prob > threshold trades.
    Returns NaN if no valid folds or no trades selected in any fold.
    """
    from lightgbm import LGBMClassifier

    folds = purged_walk_forward_cv(entry_ts, exit_ts, n_folds=5, embargo_h=12)
    if not folds:
        return float("nan")

    fold_returns = []
    for train_idx, test_idx in folds:
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        sr_te = spread_returns[test_idx]

        if y_tr.nunique() < 2 or len(X_te) == 0:
            continue

        clf = LGBMClassifier(**lgbm_params)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]

        selected = probs > threshold
        if selected.sum() == 0:
            continue

        net = sr_te[selected] - fee_per_rt
        fold_returns.append(float(np.mean(net)))

    return float(np.mean(fold_returns)) if fold_returns else float("nan")


def run_grid_search(
    feature_version: str,
    label_version: str,
) -> dict:
    """
    Run the full 72-combo grid search.
    Returns dict with 'results' (all combos) and 'best' (highest cv_net_return).
    """
    logger.info("Loading pair_features (fv=%s)...", feature_version)
    df_pf = _fetch_pair_features(feature_version)
    if df_pf.empty:
        logger.error("No pair_features data — run pipeline Steps 5–7 first")
        return {}

    # Generate valid TB pairs (stop_z must be > entry_z)
    tb_grid = [
        (ez, sz) for ez, sz in product(TB_ENTRY_Z_GRID, TB_STOP_Z_GRID)
        if sz > ez
    ]
    lgbm_grid = list(product(N_ESTIMATORS_GRID, MAX_DEPTH_GRID))
    total = len(tb_grid) * len(lgbm_grid)
    logger.info(
        "Grid: %d TB pairs × %d LightGBM combos = %d total",
        len(tb_grid), len(lgbm_grid), total,
    )

    results = []
    done = 0

    for entry_z, stop_z in tb_grid:
        # Generate TB labels in memory (not saved to DB)
        label_rows = compute_labels(
            df_pf,
            entry_z=entry_z, exit_z=0.5, stop_z=stop_z, max_hold_h=12,
        )
        if not label_rows:
            done += len(lgbm_grid)
            continue

        df_train = _build_training_df(label_rows, df_pf)
        if df_train.empty or len(df_train) < 20:
            done += len(lgbm_grid)
            continue

        X, _ = build_feature_matrix(df_train)
        if X.empty:
            done += len(lgbm_grid)
            continue

        df_train = df_train.iloc[X.index].reset_index(drop=True)
        X = X.reset_index(drop=True)
        y = (df_train["tb_label"] == 1).astype(int)
        spread_returns = df_train["spread_return"].to_numpy(dtype=float)
        entry_ts = pd.to_datetime(df_train["entry_timestamp"], utc=True).reset_index(drop=True)
        exit_ts = pd.to_datetime(df_train["exit_timestamp"], utc=True).reset_index(drop=True)

        n_labels = len(df_train)
        pos_rate = float(y.mean())

        for n_est, depth in lgbm_grid:
            params = {
                **_BASE_LGBM_PARAMS,
                "n_estimators": n_est,
                "max_depth": depth,
                "num_leaves": min(2 ** depth - 1, 31),
            }

            cv_ret = _cv_net_return(X, y, spread_returns, entry_ts, exit_ts, params)

            results.append({
                "entry_z": entry_z,
                "stop_z": stop_z,
                "n_estimators": n_est,
                "max_depth": depth,
                "n_labels": n_labels,
                "pos_rate": round(pos_rate, 4),
                "cv_net_return": round(cv_ret, 6) if not np.isnan(cv_ret) else None,
            })

            done += 1
            if done % 10 == 0 or done == total:
                logger.info(
                    "  [%d/%d] entry_z=%.1f stop_z=%.1f n_est=%d depth=%d "
                    "n_labels=%d cv_net=%.5f",
                    done, total, entry_z, stop_z, n_est, depth,
                    n_labels, cv_ret if not np.isnan(cv_ret) else -999,
                )

    # Find best by cv_net_return
    valid = [r for r in results if r["cv_net_return"] is not None]
    best = max(valid, key=lambda r: r["cv_net_return"]) if valid else None

    if best:
        logger.info(
            "Best params: entry_z=%.1f stop_z=%.1f n_est=%d depth=%d "
            "→ cv_net_return=%.5f",
            best["entry_z"], best["stop_z"], best["n_estimators"],
            best["max_depth"], best["cv_net_return"],
        )
    else:
        logger.warning("No valid grid results found")

    out = {
        "feature_version": feature_version,
        "label_version": label_version,
        "grid": {
            "entry_z": TB_ENTRY_Z_GRID,
            "stop_z": TB_STOP_Z_GRID,
            "n_estimators": N_ESTIMATORS_GRID,
            "max_depth": MAX_DEPTH_GRID,
        },
        "n_combos": total,
        "results": results,
        "best": best,
    }

    out_path = MODELS_DIR / f"hparam_results_{label_version}.json"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Grid search results saved → %s", out_path)

    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(
        description="Hyperparameter grid search for TB + LightGBM parameters"
    )
    parser.add_argument("--feature-version", default=FEATURE_VERSION, dest="feature_version")
    parser.add_argument("--label-version", default=LABEL_VERSION, dest="label_version")
    args = parser.parse_args()

    run_grid_search(args.feature_version, args.label_version)


if __name__ == "__main__":
    main()
