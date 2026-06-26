"""
DB connection pool and UPSERT helpers.
All SQL goes through this module — no raw psycopg2 elsewhere.
"""
import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import execute_values

from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        logger.info("DB pool created (%s:%s/%s)", DB_HOST, DB_PORT, DB_NAME)
    return _pool


@contextmanager
def get_conn() -> Iterator[psycopg2.extensions.connection]:
    conn = _get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _get_pool().putconn(conn)


# ─────────────────────────────────────────────────────────────────────────────
# market_data UPSERT
# ─────────────────────────────────────────────────────────────────────────────
_MARKET_DATA_COLS = (
    "symbol", "market_type", "timestamp", "interval",
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades_count", "taker_buy_volume", "source",
)

_MARKET_DATA_SQL = f"""
    INSERT INTO market_data ({", ".join(_MARKET_DATA_COLS)})
    VALUES %s
    ON CONFLICT (symbol, market_type, interval, timestamp) DO NOTHING
"""


def upsert_market_data(rows: list[dict]) -> int:
    """UPSERT rows into market_data. Returns number of rows inserted."""
    if not rows:
        return 0
    tuples = [tuple(r.get(c) for c in _MARKET_DATA_COLS) for r in rows]
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, _MARKET_DATA_SQL, tuples)
            return cur.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# open_interest UPSERT
# ─────────────────────────────────────────────────────────────────────────────
_OI_SQL = """
    INSERT INTO open_interest (symbol, timestamp, open_interest, oi_value_usdt)
    VALUES %s
    ON CONFLICT (symbol, timestamp) DO NOTHING
"""


def upsert_open_interest(rows: list[dict]) -> int:
    if not rows:
        return 0
    tuples = [(r["symbol"], r["timestamp"], r["open_interest"], r.get("oi_value_usdt")) for r in rows]
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, _OI_SQL, tuples)
            return cur.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# pair_features UPSERT
# ─────────────────────────────────────────────────────────────────────────────
_PAIR_FEATURES_COLS = (
    "pair_id", "interval", "timestamp",
    "log_price_y", "log_price_x", "hedge_beta",
    "spread", "spread_mean", "spread_std", "zscore",
    "feature_version",
)

_PAIR_FEATURES_SQL = f"""
    INSERT INTO pair_features ({", ".join(_PAIR_FEATURES_COLS)})
    VALUES %s
    ON CONFLICT (pair_id, interval, timestamp, feature_version) DO NOTHING
"""


def upsert_pair_features(rows: list[dict]) -> int:
    if not rows:
        return 0
    tuples = [tuple(r.get(c) for c in _PAIR_FEATURES_COLS) for r in rows]
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(cur, _PAIR_FEATURES_SQL, tuples)
            return cur.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# funding_rate UPDATE (forward-fill onto perp candles)
# ─────────────────────────────────────────────────────────────────────────────
_FUNDING_UPDATE_SQL = """
    UPDATE market_data
    SET    funding_rate = %(rate)s
    WHERE  symbol = %(symbol)s
      AND  market_type = 'perp'
      AND  interval    = %(interval)s
      AND  timestamp   >= %(ts_from)s
      AND  timestamp   <  %(ts_to)s
      AND  funding_rate IS NULL
"""


def update_funding_rates(records: list[dict]) -> int:
    """records: [{symbol, rate, ts_from, ts_to, interval}]"""
    if not records:
        return 0
    total = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in records:
                cur.execute(_FUNDING_UPDATE_SQL, r)
                total += cur.rowcount
    return total


# ─────────────────────────────────────────────────────────────────────────────
# realized_vol UPDATE
# ─────────────────────────────────────────────────────────────────────────────
_VOL_UPDATE_SQL = """
    UPDATE market_data
    SET    realized_vol = data.vol
    FROM   (VALUES %s) AS data(symbol, market_type, interval, ts, vol)
    WHERE  market_data.symbol      = data.symbol
      AND  market_data.market_type = data.market_type
      AND  market_data.interval    = data.interval
      AND  market_data.timestamp   = data.ts
"""


def update_realized_vol(rows: list[dict]) -> int:
    """rows: [{symbol, market_type, interval, timestamp, realized_vol}]"""
    if not rows:
        return 0
    tuples = [
        (r["symbol"], r["market_type"], r["interval"], r["timestamp"], r["realized_vol"])
        for r in rows
    ]
    with get_conn() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur, _VOL_UPDATE_SQL, tuples,
                template="(%s, %s, %s, %s::timestamptz, %s::numeric)"
            )
            return cur.rowcount
