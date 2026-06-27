"""
Phase 3 — LightGBM meta-labeling trainer (standalone script).

Loads ml_labels JOIN pair_features, runs purged walk-forward CV,
trains a final model on the full dataset, and saves it to models/.

Usage:
    python -m src.ml.train
    python -m src.ml.train --label-version v1.0-tb --feature-version v0.3-kalman
"""
import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_VERSION, LABEL_VERSION
from src.db import load_labels_for_training
from src.ml.features import build_feature_matrix
from src.ml.cv import purged_walk_forward_cv

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 300,
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


def _binary_label(tb_label: pd.Series) -> pd.Series:
    """Convert tb_label (+1/-1/0) → binary: 1 = profit hit, 0 = stop or timeout."""
    return (tb_label == 1).astype(int)


def train(feature_version: str, label_version: str,
          n_estimators: int = 300, max_depth: int = 4) -> None:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, precision_score, recall_score

    params = {**_LGBM_PARAMS, "n_estimators": n_estimators, "max_depth": max_depth}

    logger.info("Loading training data (fv=%s, lv=%s)", feature_version, label_version)
    df = load_labels_for_training(feature_version, label_version)
    if df.empty:
        logger.error("No training data found — run Step 8 (triple-barrier labeling) first")
        return

    X, feat_names = build_feature_matrix(df)
    # Re-align df to X's index after NaN drop
    df = df.iloc[X.index]
    y = _binary_label(df["tb_label"])
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    entry_ts = pd.to_datetime(df["entry_timestamp"].reset_index(drop=True), utc=True)
    exit_ts = pd.to_datetime(df["exit_timestamp"].reset_index(drop=True), utc=True)

    logger.info("Dataset: %d samples, %d features, %.1f%% positive",
                len(y), len(feat_names), 100 * y.mean())

    # ── Purged walk-forward CV ────────────────────────────────────────────────
    folds = purged_walk_forward_cv(entry_ts, exit_ts, n_folds=5, embargo_h=12)
    logger.info("Purged CV: %d usable folds", len(folds))

    cv_scores = {"roc_auc": [], "precision": [], "recall": []}
    for k, (train_idx, test_idx) in enumerate(folds):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        if y_tr.nunique() < 2:
            logger.warning("Fold %d: single class in train — skipping", k)
            continue

        clf = LGBMClassifier(**params)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]

        auc = roc_auc_score(y_te, prob)
        pred = (prob > 0.5).astype(int)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)

        cv_scores["roc_auc"].append(auc)
        cv_scores["precision"].append(prec)
        cv_scores["recall"].append(rec)
        logger.info(
            "  Fold %d | train=%d test=%d | AUC=%.3f prec=%.3f rec=%.3f",
            k, len(train_idx), len(test_idx), auc, prec, rec,
        )

    if cv_scores["roc_auc"]:
        logger.info(
            "CV summary | AUC=%.3f±%.3f prec=%.3f±%.3f rec=%.3f±%.3f",
            np.mean(cv_scores["roc_auc"]), np.std(cv_scores["roc_auc"]),
            np.mean(cv_scores["precision"]), np.std(cv_scores["precision"]),
            np.mean(cv_scores["recall"]), np.std(cv_scores["recall"]),
        )

    # ── Final model on full data ──────────────────────────────────────────────
    logger.info("Training final model on full dataset (%d samples)", len(y))
    final_clf = LGBMClassifier(**params)
    final_clf.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_stem = f"meta_lgbm_{label_version}"
    model_path = MODELS_DIR / f"{model_stem}.txt"
    meta_path = MODELS_DIR / f"{model_stem}.json"

    final_clf.booster_.save_model(str(model_path))
    logger.info("Model saved → %s", model_path)

    meta = {
        "label_version": label_version,
        "feature_version": feature_version,
        "n_samples": int(len(y)),
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "positive_rate": float(y.mean()),
        "cv_folds": len(cv_scores["roc_auc"]),
        "cv_roc_auc_mean": float(np.mean(cv_scores["roc_auc"])) if cv_scores["roc_auc"] else None,
        "cv_roc_auc_std": float(np.std(cv_scores["roc_auc"])) if cv_scores["roc_auc"] else None,
        "lgbm_params": params,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    logger.info("Metadata saved → %s", meta_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Train LightGBM meta-labeling model")
    parser.add_argument("--feature-version", default=FEATURE_VERSION, dest="feature_version")
    parser.add_argument("--label-version", default=LABEL_VERSION, dest="label_version")
    parser.add_argument("--n-estimators", type=int, default=300, dest="n_estimators")
    parser.add_argument("--max-depth", type=int, default=4, dest="max_depth")
    args = parser.parse_args()

    train(args.feature_version, args.label_version,
          n_estimators=args.n_estimators, max_depth=args.max_depth)


if __name__ == "__main__":
    main()
