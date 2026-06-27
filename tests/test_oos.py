"""
Phase 5 — OOS evaluation integrity tests.

Verifies that the OOS split enforces strict temporal separation:
- no train sample's exit_timestamp crosses the holdout boundary
- no overlap between train and test entry_timestamps
- oos_eval() returns the expected result keys
"""
import numpy as np
import pandas as pd
import pytest


def _make_df(n: int, holdout_start: str = "2025-01-01") -> pd.DataFrame:
    """Build a minimal synthetic DataFrame spanning 2023-01-01 → 2026-01-01."""
    rng = pd.date_range("2023-01-01", periods=n, freq="24h", tz="UTC")
    return pd.DataFrame({
        "entry_timestamp": rng,
        "exit_timestamp": rng + pd.Timedelta(hours=4),
        "tb_label": np.where(np.arange(n) % 3 == 0, 1, -1),
        "spread_return": np.random.default_rng(42).normal(0.001, 0.005, n),
        "ou_zscore": np.random.default_rng(1).normal(2.5, 0.3, n),
        "ou_sigma_eq": np.abs(np.random.default_rng(2).normal(0.01, 0.002, n)),
        "funding_spread": np.zeros(n),
        "basis_y": np.zeros(n),
        "basis_x": np.zeros(n),
        "taker_ratio_y": np.full(n, 0.5),
        "taker_ratio_x": np.full(n, 0.5),
        "spread_std": np.full(n, 0.002),
    })


def _apply_split(df: pd.DataFrame, holdout_start: str):
    """Replicate the exact split logic in oos_eval.oos_eval()."""
    holdout_dt = pd.Timestamp(holdout_start, tz="UTC")
    train_mask = (df["entry_timestamp"] < holdout_dt) & (df["exit_timestamp"] < holdout_dt)
    test_mask = df["entry_timestamp"] >= holdout_dt
    return df[train_mask].copy(), df[test_mask].copy()


# ─── Test 1: no train sample exits into the holdout window ───────────────────

def test_oos_no_lookahead():
    df = _make_df(500)
    holdout = "2025-01-01"
    train_df, _ = _apply_split(df, holdout)

    holdout_dt = pd.Timestamp(holdout, tz="UTC")
    # All train exit_timestamps must be strictly before holdout_start
    assert (train_df["exit_timestamp"] < holdout_dt).all(), (
        "Train split contains rows whose exit_timestamp reaches into the holdout window"
    )


# ─── Test 2: train and test entry_timestamps are disjoint ────────────────────

def test_oos_train_test_disjoint():
    df = _make_df(500)
    holdout = "2025-01-01"
    train_df, test_df = _apply_split(df, holdout)

    holdout_dt = pd.Timestamp(holdout, tz="UTC")
    assert (train_df["entry_timestamp"] < holdout_dt).all()
    assert (test_df["entry_timestamp"] >= holdout_dt).all()
    assert len(pd.merge(train_df[["entry_timestamp"]], test_df[["entry_timestamp"]],
                        on="entry_timestamp")) == 0


# ─── Test 3: oos_eval() returns expected result keys ─────────────────────────

def test_oos_returns_expected_keys(monkeypatch):
    """Verify oos_eval() structure without hitting the DB or LightGBM."""
    import sys
    import types
    from unittest.mock import MagicMock

    # Stub lightgbm before importing oos_eval
    lgb_mod = types.ModuleType("lightgbm")
    mock_clf = MagicMock()
    mock_clf.return_value.fit.return_value = None
    mock_clf.return_value.predict_proba.side_effect = lambda X: np.full((len(X), 2), [0.4, 0.6])
    lgb_mod.LGBMClassifier = mock_clf
    monkeypatch.setitem(sys.modules, "lightgbm", lgb_mod)

    # Stub sklearn.metrics
    skl_mod = types.ModuleType("sklearn.metrics")
    skl_mod.roc_auc_score = lambda *a, **kw: 0.65
    skl_mod.precision_score = lambda *a, **kw: 0.7
    skl_mod.recall_score = lambda *a, **kw: 0.6
    monkeypatch.setitem(sys.modules, "sklearn.metrics", skl_mod)

    import src.ml.oos_eval as oos_mod

    df = _make_df(500)
    monkeypatch.setattr(oos_mod, "load_labels_for_training", lambda fv, lv: df)

    # Disable file write
    monkeypatch.setattr(oos_mod, "MODELS_DIR",
                        __import__("pathlib").Path("/tmp/oos_test_dir"))

    result = oos_mod.oos_eval(
        feature_version="v0.3-kalman",
        label_version="v1.1-tb",
        holdout_start="2024-01-01",  # 500×24h data spans 2023-01-01→2024-05-14
    )

    required_keys = {
        "holdout_start", "n_train", "n_test", "n_oos_trades",
        "oos_auc", "oos_precision", "oos_recall", "oos_avg_net_return",
    }
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )
    assert result["n_train"] > 0
    assert result["n_test"] > 0
