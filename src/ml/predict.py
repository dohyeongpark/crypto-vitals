"""
Phase 3 — LightGBM meta-labeling batch inference.

Finds ml_labels rows where meta_prob IS NULL, builds feature matrix from
pair_features at entry_timestamp, runs model.predict_proba, and writes
meta_prob + meta_label back to the table.

Usage:
    python -m src.ml.predict
    python -m src.ml.predict --label-version v1.0-tb --feature-version v0.3-kalman
"""
import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import FEATURE_VERSION, LABEL_VERSION, META_THRESHOLD
from src.db import get_engine, update_meta_predictions
from src.ml.features import build_feature_matrix

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _load_unscored(feature_version: str, label_version: str) -> pd.DataFrame:
    sql = """
        SELECT
            ml.entry_timestamp, ml.exit_timestamp,
            pf.ou_zscore, pf.ou_halflife, pf.ou_kappa, pf.ou_sigma_eq, pf.ou_mu,
            pf.eg_pvalue, pf.johansen_trace,
            pf.basis_y, pf.basis_x,
            pf.taker_ratio_y, pf.taker_ratio_x,
            pf.funding_spread, pf.spread_std, pf.spread_kalman
        FROM ml_labels ml
        JOIN pair_features pf
          ON pf.timestamp       = ml.entry_timestamp
         AND pf.feature_version = ml.feature_version
        WHERE ml.feature_version = %(fv)s
          AND ml.label_version   = %(lv)s
          AND ml.meta_prob       IS NULL
        ORDER BY ml.entry_timestamp
    """
    return pd.read_sql_query(
        sql, get_engine(),
        params={"fv": feature_version, "lv": label_version},
        parse_dates=["entry_timestamp", "exit_timestamp"],
    )


def predict(
    feature_version: str,
    label_version: str,
    threshold: float = META_THRESHOLD,
) -> int:
    """Run batch inference; returns number of rows updated."""
    import lightgbm as lgb

    model_path = MODELS_DIR / f"meta_lgbm_{label_version}.txt"
    if not model_path.exists():
        logger.error("Model not found at %s — run src.ml.train first", model_path)
        return 0

    model = lgb.Booster(model_file=str(model_path))
    logger.info("Loaded model from %s", model_path)

    df = _load_unscored(feature_version, label_version)
    if df.empty:
        logger.info("No unscored ml_labels rows found — nothing to do")
        return 0

    logger.info("Scoring %d rows", len(df))
    X, _ = build_feature_matrix(df)
    if X.empty:
        logger.warning("All rows dropped after NaN filtering — check feature data")
        return 0

    probs = model.predict(X.values)
    labels = (probs > threshold).astype(int)

    # Re-align entry_timestamps to X after NaN drop
    scored_ts = pd.to_datetime(df["entry_timestamp"].iloc[X.index], utc=True)

    rows = [
        {
            "entry_timestamp": ts.to_pydatetime(),
            "meta_prob": float(prob),
            "meta_label": int(label),
        }
        for ts, prob, label in zip(scored_ts, probs, labels)
    ]

    n = update_meta_predictions(rows, feature_version, label_version)
    logger.info("Updated %d ml_labels rows with meta_prob/meta_label", n)
    return n


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Batch LightGBM meta-labeling inference")
    parser.add_argument("--feature-version", default=FEATURE_VERSION, dest="feature_version")
    parser.add_argument("--label-version", default=LABEL_VERSION, dest="label_version")
    parser.add_argument("--threshold", type=float, default=META_THRESHOLD)
    args = parser.parse_args()

    predict(args.feature_version, args.label_version, threshold=args.threshold)


if __name__ == "__main__":
    main()
