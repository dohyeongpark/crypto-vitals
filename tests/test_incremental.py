"""
Phase 5 — Incremental pipeline idempotency tests.

Verifies that running a step with incremental=True when no new data
exists produces zero new insertions (idempotency).
"""
import pandas as pd
import pytest


# ─── Test 1: triple_barrier incremental with no new data ─────────────────────

def test_triple_barrier_incremental_idempotent(monkeypatch):
    """
    When max_entry_ts is at the last pair_features row and there are no newer
    rows, _fetch_features(since=max_entry_ts) returns an empty DataFrame →
    compute_and_store returns 0 (nothing to label).
    """
    import src.labels.triple_barrier as tb_mod

    last_ts = pd.Timestamp("2025-06-01 00:00:00", tz="UTC")
    empty_df = pd.DataFrame(columns=["timestamp", "ou_zscore", "spread_kalman"])

    monkeypatch.setattr(tb_mod, "_fetch_max_entry_ts", lambda lv: last_ts)
    # In incremental mode, since=last_ts → no rows newer than that → empty
    monkeypatch.setattr(tb_mod, "_fetch_features",
                        lambda fv, since=None: empty_df if since is not None else empty_df)
    monkeypatch.setattr(tb_mod, "upsert_ml_labels", lambda rows: len(rows))

    n = tb_mod.compute_and_store(
        "v0.3-kalman", "v1.1-tb",
        entry_z=2.5, stop_z=3.5,
        incremental=True,
    )
    assert n == 0


# ─── Test 2: pair_spread incremental fetch_since logic ───────────────────────

def test_pair_spread_incremental_fetch_since(monkeypatch):
    """
    In incremental mode, _fetch_spot_closes should be called with a since
    timestamp equal to max_ts - 2×window.
    """
    import src.features.pair_spread as ps_mod

    window = 168
    max_ts = pd.Timestamp("2025-06-01 00:00:00", tz="UTC")
    expected_since = max_ts - pd.Timedelta(hours=window * 2)

    captured = {}

    monkeypatch.setattr(ps_mod, "_fetch_max_processed_ts", lambda version: max_ts)
    monkeypatch.setattr(
        ps_mod, "_fetch_spot_closes",
        lambda since=None: captured.update({"since": since}) or pd.DataFrame(),
    )

    ps_mod.compute_and_store(window, "v0.3-kalman", incremental=True)

    assert captured["since"] == expected_since
