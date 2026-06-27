"""
Phase 5 — Strict temporal out-of-sample evaluation.

Trains on data before holdout_start, evaluates on data from holdout_start onward.
Purges train samples whose exit_timestamp reaches into the holdout window to
prevent any lookahead across the boundary.

Usage:
    python -m src.ml.oos_eval
    python -m src.ml.oos_eval --label-version v1.1-tb --holdout-start 2025-01-01
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_VERSION, LABEL_VERSION, META_THRESHOLD
from src.db import load_labels_for_training
from src.ml.features import build_feature_matrix

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 200,
    "max_depth": 4,
    "num_leaves": 15,
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

FEE_PER_RT = 0.006  # 0.4% fee + 0.2% slippage


def oos_eval(
    feature_version: str = FEATURE_VERSION,
    label_version: str = LABEL_VERSION,
    holdout_start: str = "2025-01-01",
    meta_threshold: float = META_THRESHOLD,
    fee_per_rt: float = FEE_PER_RT,
    n_estimators: int = 200,
    max_depth: int = 4,
) -> dict:
    """
    Train on entry_timestamp < holdout_start, evaluate on >= holdout_start.
    Purges train samples where exit_timestamp >= holdout_start.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import precision_score, recall_score, roc_auc_score

    df = load_labels_for_training(feature_version, label_version)
    if df.empty:
        logger.error("No training data — run triple-barrier labeling first")
        return {}

    holdout_dt = pd.Timestamp(holdout_start, tz="UTC")

    train_mask = (df["entry_timestamp"] < holdout_dt) & (df["exit_timestamp"] < holdout_dt)
    test_mask = df["entry_timestamp"] >= holdout_dt

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(
        "OOS split: train=%d (purged %d boundary), test=%d (holdout_start=%s)",
        len(train_df),
        int((df["entry_timestamp"] < holdout_dt).sum()) - len(train_df),
        len(test_df),
        holdout_start,
    )

    if train_df.empty:
        logger.error("No training data before %s", holdout_start)
        return {}
    if test_df.empty:
        logger.warning("No test data from %s onward — OOS window may be empty", holdout_start)
        return {"n_train": len(train_df), "n_test": 0}

    X_train, feat_names = build_feature_matrix(train_df)
    train_df = train_df.iloc[X_train.index].reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = (train_df["tb_label"] == 1).astype(int)

    X_test, _ = build_feature_matrix(test_df)
    test_df = test_df.iloc[X_test.index].reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_test = (test_df["tb_label"] == 1).astype(int)

    params = {**_LGBM_PARAMS, "n_estimators": n_estimators, "max_depth": max_depth}
    clf = LGBMClassifier(**params)
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)[:, 1]
    pred = (prob > meta_threshold).astype(int)

    auc = float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else float("nan")
    prec = float(precision_score(y_test, pred, zero_division=0))
    rec = float(recall_score(y_test, pred, zero_division=0))

    selected = prob > meta_threshold
    oos_returns = test_df["spread_return"].values[selected]
    avg_net = float((oos_returns - fee_per_rt).mean()) if selected.any() else float("nan")

    n_pos = int(y_test.sum())
    n_neg = int((y_test == 0).sum())

    result = {
        "holdout_start": holdout_start,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_test_pos": n_pos,
        "n_test_neg": n_neg,
        "n_oos_trades": int(selected.sum()),
        "oos_auc": round(auc, 4),
        "oos_precision": round(prec, 4),
        "oos_recall": round(rec, 4),
        "oos_avg_net_return": round(avg_net, 6) if not np.isnan(avg_net) else None,
        "fee_per_rt": fee_per_rt,
        "meta_threshold": meta_threshold,
        "feature_version": feature_version,
        "label_version": label_version,
        "n_features": len(feat_names),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"oos_{label_version}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("OOS results saved → %s", out_path)

    logger.info(
        "OOS results | n_train=%d n_test=%d n_trades=%d | "
        "AUC=%.3f prec=%.3f rec=%.3f avg_net=%.4f%%",
        result["n_train"], result["n_test"], result["n_oos_trades"],
        auc, prec, rec,
        avg_net * 100 if not np.isnan(avg_net) else float("nan"),
    )
    return result


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Strict temporal OOS evaluation")
    parser.add_argument("--feature-version", default=FEATURE_VERSION, dest="feature_version")
    parser.add_argument("--label-version", default=LABEL_VERSION, dest="label_version")
    parser.add_argument("--holdout-start", default="2025-01-01", dest="holdout_start")
    parser.add_argument("--threshold", type=float, default=META_THRESHOLD, dest="meta_threshold")
    parser.add_argument("--n-estimators", type=int, default=200, dest="n_estimators")
    parser.add_argument("--max-depth", type=int, default=4, dest="max_depth")
    args = parser.parse_args()

    oos_eval(
        feature_version=args.feature_version,
        label_version=args.label_version,
        holdout_start=args.holdout_start,
        meta_threshold=args.meta_threshold,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
