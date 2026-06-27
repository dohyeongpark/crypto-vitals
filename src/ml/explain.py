"""
Phase 4 — SHAP feature importance for the LightGBM meta-labeling model.

Uses TreeExplainer (exact, no sampling) on the full training dataset.
Saves mean |SHAP| per feature to models/shap_{label_version}.json.

Usage:
    python -m src.ml.explain
    python -m src.ml.explain --label-version v1.0-tb --feature-version v0.3-kalman
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.config import FEATURE_VERSION, LABEL_VERSION
from src.db import load_labels_for_training
from src.ml.features import build_feature_matrix

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def explain(
    feature_version: str,
    label_version: str,
    max_display: int = 10,
) -> dict:
    """
    Compute SHAP feature importance for the trained meta-labeling model.

    Returns {feature_name: mean_abs_shap} sorted descending.
    Saves results to models/shap_{label_version}.json.
    """
    import lightgbm as lgb
    import shap

    model_path = MODELS_DIR / f"meta_lgbm_{label_version}.txt"
    if not model_path.exists():
        logger.error("Model not found at %s — run src.ml.train first", model_path)
        return {}

    model = lgb.Booster(model_file=str(model_path))
    logger.info("Loaded model from %s", model_path)

    df = load_labels_for_training(feature_version, label_version)
    if df.empty:
        logger.error("No training data found for fv=%s lv=%s", feature_version, label_version)
        return {}

    X, feat_names = build_feature_matrix(df)
    logger.info("Computing SHAP values (N=%d, features=%d)...", len(X), len(feat_names))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.values)   # shape (N, n_features)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = dict(sorted(
        zip(feat_names, mean_abs_shap.tolist()),
        key=lambda x: -x[1],
    ))

    # Save to JSON
    out_path = MODELS_DIR / f"shap_{label_version}.json"
    out_path.write_text(json.dumps(importance, indent=2))
    logger.info("SHAP importance saved → %s", out_path)

    # Log top-N
    logger.info("Top-%d features by mean |SHAP|:", max_display)
    for rank, (feat, val) in enumerate(list(importance.items())[:max_display], 1):
        logger.info("  %2d. %-20s  %.6f", rank, feat, val)

    return importance


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="SHAP feature importance for meta-labeling model")
    parser.add_argument("--feature-version", default=FEATURE_VERSION, dest="feature_version")
    parser.add_argument("--label-version", default=LABEL_VERSION, dest="label_version")
    parser.add_argument("--top", type=int, default=10, dest="max_display")
    args = parser.parse_args()

    explain(args.feature_version, args.label_version, max_display=args.max_display)


if __name__ == "__main__":
    main()
