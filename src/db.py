"""
DB connection pool and UPSERT helpers.
All SQL goes through this module — no raw psycopg2 elsewhere.

Connection strategy:
  get_conn()   — psycopg2 ThreadedConnectionPool, used for all writes (UPSERT/UPDATE)
  get_engine() — SQLAlchemy engine, used for reads via pd.read_sql_query
"""
import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import execute_values
from sqlalchemy import create_engine as _sa_create_engine

from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None
_engine = None


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


def get_engine():
    """SQLAlchemy engine for pd.read_sql_query (read-only analytics)."""
    global _engine
    if _engine is None:
        url = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        _engine = _sa_create_engine(url, pool_pre_ping=True)
    return _engine


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
    RETURNING 1
"""


def upsert_pair_features(rows: list[dict]) -> int:
    if not rows:
        return 0
    tuples = [tuple(r.get(c) for c in _PAIR_FEATURES_COLS) for r in rows]
    with get_conn() as conn:
        with conn.cursor() as cur:
            # fetch=True collects RETURNING rows; len() gives actual insert count
            # (ON CONFLICT DO NOTHING skips conflicts, RETURNING omits them)
            result = execute_values(cur, _PAIR_FEATURES_SQL, tuples, fetch=True)
            return len(result)


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


# ─────────────────────────────────────────────────────────────────────────────
# regime features UPDATE (batch-fill new columns on existing pair_features rows)
# ─────────────────────────────────────────────────────────────────────────────

_REGIME_COLS = (
    "funding_rate_y", "funding_rate_x", "funding_spread",
    "taker_ratio_y", "taker_ratio_x",
    "basis_y", "basis_x",
)

_REGIME_UPDATE_SQL_TMPL = """
    UPDATE pair_features AS pf
    SET funding_rate_y = d.funding_rate_y::numeric,
        funding_rate_x = d.funding_rate_x::numeric,
        funding_spread = d.funding_spread::numeric,
        taker_ratio_y  = d.taker_ratio_y::numeric,
        taker_ratio_x  = d.taker_ratio_x::numeric,
        basis_y        = d.basis_y::numeric,
        basis_x        = d.basis_x::numeric
    FROM (VALUES %s) AS d(
        ts,
        funding_rate_y, funding_rate_x, funding_spread,
        taker_ratio_y, taker_ratio_x,
        basis_y, basis_x
    )
    WHERE pf.timestamp       = d.ts::timestamptz
      AND pf.pair_id         = 'ETHUSDT_BTCUSDT'
      AND pf.feature_version = '{{version}}'
    RETURNING 1
"""


def update_regime_features(rows: list[dict], version: str) -> int:
    """rows: [{timestamp, funding_rate_y, funding_rate_x, funding_spread,
               taker_ratio_y, taker_ratio_x, basis_y, basis_x}]"""
    if not rows:
        return 0
    tuples = [
        (r["timestamp"], *(r.get(c) for c in _REGIME_COLS))
        for r in rows
    ]
    # Format version into SQL (internal config value, not user input)
    sql = _REGIME_UPDATE_SQL_TMPL.replace("{{version}}", version)
    with get_conn() as conn:
        with conn.cursor() as cur:
            # fetch=True accumulates RETURNING rows across all internal pages,
            # giving an accurate total count regardless of page_size.
            result = execute_values(cur, sql, tuples, fetch=True)
            return len(result)
