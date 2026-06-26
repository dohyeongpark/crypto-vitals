"""
Track A-2: Fetch historical funding rates and forward-fill onto perp candles.

Binance FAPI /fapi/v1/fundingRate — paginated (limit ≤ 1000, 8h intervals).
Funding applies from its fundingTime until the NEXT fundingTime.
Forward-fill assigns each perp 1h candle the most recent funding rate ≤ timestamp.

Usage:
    python -m src.collectors.funding --start 2023-01-01
    python -m src.collectors.funding --symbols BTCUSDT --start 2024-01-01 --end 2024-06-01
"""
import argparse
import logging
import time
from datetime import datetime, timezone

import requests

from src.config import (
    BINANCE_FAPI_BASE, HTTP_BACKOFF_BASE, HTTP_MAX_RETRIES, HTTP_TIMEOUT, INTERVAL, SYMBOLS,
)
from src.db import update_funding_rates

logger = logging.getLogger(__name__)

_FUNDING_URL = f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate"
_PAGE_LIMIT = 1000


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get(params: dict) -> list[dict]:
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            resp = requests.get(_FUNDING_URL, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            delay = HTTP_BACKOFF_BASE * (2 ** attempt)
            if attempt == HTTP_MAX_RETRIES:
                raise
            logger.warning("Retry %d/%d in %.1fs: %s", attempt, HTTP_MAX_RETRIES, delay, exc)
            time.sleep(delay)
    return []


# ── Pagination ────────────────────────────────────────────────────────────────

def fetch_funding(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    """Fetch all funding rate records for symbol in [start_ms, end_ms]."""
    records: list[dict] = []
    cursor = start_ms

    while cursor < end_ms:
        page = _get({"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": _PAGE_LIMIT})
        if not page:
            break
        records.extend(page)
        last_ts = int(page[-1]["fundingTime"])
        if last_ts <= cursor or len(page) < _PAGE_LIMIT:
            break
        cursor = last_ts + 1

    logger.info("[%s] fetched %d funding records", symbol, len(records))
    return records


# ── Forward-fill helper ───────────────────────────────────────────────────────

def _build_fill_records(symbol: str, raw: list[dict]) -> list[dict]:
    """
    Convert raw funding records into DB update records for forward-fill.
    Each record covers [fundingTime, nextFundingTime) on perp candles.
    """
    if not raw:
        return []

    # Sort by fundingTime ascending
    sorted_raw = sorted(raw, key=lambda r: int(r["fundingTime"]))

    fill_records = []
    for i, r in enumerate(sorted_raw):
        ts_from = datetime.fromtimestamp(int(r["fundingTime"]) / 1000, tz=timezone.utc)
        # Forward-fill until the next funding time (or far future for the last record)
        if i + 1 < len(sorted_raw):
            ts_to = datetime.fromtimestamp(int(sorted_raw[i + 1]["fundingTime"]) / 1000, tz=timezone.utc)
        else:
            # Last record: fill forward indefinitely (UPDATE only touches existing rows)
            ts_to = datetime(9999, 12, 31, tzinfo=timezone.utc)

        fill_records.append({
            "symbol": symbol,
            "rate": r["fundingRate"],
            "ts_from": ts_from,
            "ts_to": ts_to,
            "interval": INTERVAL,
        })

    return fill_records


# ── Main entry ────────────────────────────────────────────────────────────────

def run(symbols: list[str], start: datetime, end: datetime) -> None:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    for symbol in symbols:
        raw = fetch_funding(symbol, start_ms, end_ms)
        fill_records = _build_fill_records(symbol, raw)
        n = update_funding_rates(fill_records)
        logger.info("[%s] updated %d perp candle rows with funding_rate", symbol, n)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Fetch Binance funding rates and forward-fill onto perp candles")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: now)")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)

    run(args.symbols, start, end)


if __name__ == "__main__":
    main()
