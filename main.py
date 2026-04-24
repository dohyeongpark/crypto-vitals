"""
Real-time cryptocurrency OHLCV collector for Binance.
Designed for GKE deployment as a long-running container.
"""

import os
import time
import logging
import json
import signal
import sys
from datetime import datetime, timezone
from collections import deque

import ccxt
import pandas as pd
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Logging — plain text with ISO-8601 timestamps, goes to stdout for K8s
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logging.Formatter.converter = time.gmtime   # UTC timestamps
logger = logging.getLogger("collector")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOLS = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME = "1m"
VOLATILITY_WINDOW = 5          # minutes of returns used for std-dev
FETCH_LIMIT = VOLATILITY_WINDOW + 2   # fetch a few extra for safety
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0         # seconds; exponential backoff

BQ_PROJECT = os.environ.get("BQ_PROJECT", "")
BQ_DATASET = os.environ.get("BQ_DATASET", "crypto_vitals")
BQ_TABLE   = os.environ.get("BQ_TABLE", "ohlcv")


# ---------------------------------------------------------------------------
# Exchange initialisation
# ---------------------------------------------------------------------------
def build_exchange() -> ccxt.binance:
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    exchange = ccxt.binance(
        {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,       # ccxt built-in throttle
            "options": {"defaultType": "spot"},
        }
    )
    logger.info("Exchange initialised (authenticated=%s)", bool(api_key))
    return exchange


# ---------------------------------------------------------------------------
# Fetch with retry + exponential backoff
# ---------------------------------------------------------------------------
def fetch_ohlcv(
    exchange: ccxt.binance,
    symbol: str,
    limit: int = FETCH_LIMIT,
) -> list[list]:
    """
    Fetch recent 1-minute candles with automatic retry on transient errors.
    Returns raw OHLCV list: [[ts, open, high, low, close, volume], ...]
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
            logger.debug("%s: fetched %d candles (attempt %d)", symbol, len(candles), attempt)
            return candles
        except ccxt.RateLimitExceeded as exc:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("%s: rate limit exceeded, retry %d/%d in %.1fs — %s",
                           symbol, attempt, MAX_RETRIES, delay, exc)
            time.sleep(delay)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning("%s: network error, retry %d/%d in %.1fs — %s",
                           symbol, attempt, MAX_RETRIES, delay, exc)
            time.sleep(delay)
        except ccxt.ExchangeError as exc:
            logger.error("%s: unrecoverable exchange error — %s", symbol, exc)
            raise
    raise RuntimeError(f"{symbol}: exhausted {MAX_RETRIES} fetch retries")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_candles(candles: list[list], symbol: str) -> bool:
    if not candles:
        logger.error("%s: empty candle list", symbol)
        return False

    for row in candles:
        ts, o, h, l, c, v = row
        if None in (ts, o, h, l, c, v):
            logger.error("%s: null field in candle %s", symbol, row)
            return False
        if not (h >= o and h >= c and l <= o and l <= c and h >= l):
            logger.error("%s: OHLC sanity check failed: %s", symbol, row)
            return False
        if v < 0:
            logger.error("%s: negative volume in candle %s", symbol, row)
            return False

    logger.debug("%s: %d candles passed validation", symbol, len(candles))
    return True


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
    """
    Adds a 'volatility' column: rolling std-dev of log returns over `window` periods.
    NaN for rows with insufficient history.
    """
    df = df.copy()
    df["log_return"] = (df["close"] / df["close"].shift(1)).apply(
        lambda x: x if pd.isna(x) else __import__("math").log(x)
    )
    df["volatility"] = df["log_return"].rolling(window).std()
    df.drop(columns=["log_return"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Build structured record
# ---------------------------------------------------------------------------
def build_record(symbol: str, df: pd.DataFrame) -> dict:
    """
    Returns the latest completed candle as a structured dict ready for
    downstream storage (DB insert, Pub/Sub message, etc.).
    """
    # The last candle in the list may still be forming; use second-to-last
    # when we have enough rows, otherwise take the last.
    row = df.iloc[-2] if len(df) >= 2 else df.iloc[-1]

    return {
        "symbol": symbol,
        "timestamp": datetime.fromtimestamp(row["timestamp"] / 1000, tz=timezone.utc).isoformat(),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
        "volatility_5m": None if pd.isna(row["volatility"]) else float(row["volatility"]),
        "collected_at": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Per-symbol pipeline
# ---------------------------------------------------------------------------
def process_symbol(exchange: ccxt.binance, symbol: str) -> dict | None:
    try:
        candles = fetch_ohlcv(exchange, symbol)
    except Exception as exc:
        logger.error("%s: fetch failed — %s", symbol, exc)
        return None

    if not validate_candles(candles, symbol):
        return None

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(candles, columns=columns)
    df = compute_volatility(df)

    record = build_record(symbol, df)
    logger.info(
        "%s | ts=%s close=%.2f vol_5m=%s",
        symbol,
        record["timestamp"],
        record["close"],
        f"{record['volatility_5m']:.6f}" if record["volatility_5m"] is not None else "n/a",
    )
    return record


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------
def seconds_until_next_minute() -> float:
    """Return fractional seconds until the top of the next UTC minute."""
    now = time.time()
    return 60.0 - (now % 60.0)


_bq_client: bigquery.Client | None = None


def _get_bq_client() -> bigquery.Client:
    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=BQ_PROJECT or None)
    return _bq_client


def emit_records(records: list[dict]) -> None:
    for record in records:
        logger.info(json.dumps(record, ensure_ascii=False))

    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}" if BQ_PROJECT else f"{BQ_DATASET}.{BQ_TABLE}"
    errors = _get_bq_client().insert_rows_json(table_ref, records)
    if errors:
        logger.error("BigQuery insert errors: %s", errors)
    else:
        logger.info("Inserted %d row(s) into %s", len(records), table_ref)


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_running = True


def _handle_signal(signum, _frame):
    global _running
    logger.info("Received signal %d, shutting down…", signum)
    _running = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Starting collector — symbols=%s timeframe=%s", SYMBOLS, TIMEFRAME)
    exchange = build_exchange()

    while _running:
        wait = seconds_until_next_minute()
        logger.debug("Sleeping %.1fs until next minute boundary", wait)
        time.sleep(wait)

        if not _running:
            break

        tick_start = time.monotonic()
        records = []

        for symbol in SYMBOLS:
            record = process_symbol(exchange, symbol)
            if record:
                records.append(record)

        if records:
            emit_records(records)

        elapsed = time.monotonic() - tick_start
        logger.debug("Tick completed in %.2fs", elapsed)

    logger.info("Collector stopped cleanly.")


if __name__ == "__main__":
    main()
