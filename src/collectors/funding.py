"""
Track A-2: Fetch historical funding rates from data.binance.vision CDN.

Downloads monthly ZIP files (same CDN as klines_bulk) — no API key needed,
no US geo-restriction. CSV columns: calc_time (ms), funding_interval_hours,
last_funding_rate.

Each funding rate applies from calc_time until the next calc_time (forward-fill).

Usage:
    python -m src.collectors.funding --start 2023-01
    python -m src.collectors.funding --symbols BTCUSDT --start 2024-01 --end 2024-06
"""
import argparse
import io
import logging
import time
import zipfile
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from src.config import HTTP_BACKOFF_BASE, HTTP_MAX_RETRIES, HTTP_TIMEOUT, INTERVAL, SYMBOLS
from src.db import update_funding_rates

logger = logging.getLogger(__name__)

_VISION_BASE = "https://data.binance.vision"


def _month_url(symbol: str, year: int, month: int) -> str:
    return (
        f"{_VISION_BASE}/data/futures/um/monthly/fundingRate"
        f"/{symbol}/{symbol}-fundingRate-{year}-{month:02d}.zip"
    )


def _download_month(symbol: str, year: int, month: int) -> Optional[list[dict]]:
    url = _month_url(symbol, year, month)
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
            if resp.status_code == 404:
                return None  # month not published yet
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                df = pd.read_csv(zf.open(zf.namelist()[0]), header=0)
            # Normalise to the same dict shape as the old API response
            records = [
                {"fundingTime": int(row["calc_time"]), "fundingRate": float(row["last_funding_rate"])}
                for _, row in df.iterrows()
            ]
            logger.info("[%s] %d-%02d: %d funding records", symbol, year, month, len(records))
            return records
        except requests.exceptions.RequestException as exc:
            delay = HTTP_BACKOFF_BASE * (2 ** attempt)
            if attempt == HTTP_MAX_RETRIES:
                logger.error("[%s] failed %d-%02d after %d tries: %s", symbol, year, month, attempt, exc)
                return None
            logger.warning("Retry %d/%d in %.1fs: %s", attempt, HTTP_MAX_RETRIES, delay, exc)
            time.sleep(delay)
    return None


def fetch_funding(symbol: str, start: datetime, end: datetime) -> list[dict]:
    """Download all monthly funding rate CSVs for symbol in [start, end]."""
    all_records: list[dict] = []
    year, month = start.year, start.month
    end_year, end_month = end.year, end.month

    while (year, month) <= (end_year, end_month):
        records = _download_month(symbol, year, month)
        if records:
            all_records.extend(records)
        month += 1
        if month > 12:
            month = 1
            year += 1

    logger.info("[%s] total funding records fetched: %d", symbol, len(all_records))
    return all_records


def _build_fill_records(symbol: str, raw: list[dict]) -> list[dict]:
    """Convert funding records into DB update records for forward-fill."""
    if not raw:
        return []

    sorted_raw = sorted(raw, key=lambda r: r["fundingTime"])

    fill_records = []
    for i, r in enumerate(sorted_raw):
        ts_from = datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc)
        if i + 1 < len(sorted_raw):
            ts_to = datetime.fromtimestamp(sorted_raw[i + 1]["fundingTime"] / 1000, tz=timezone.utc)
        else:
            ts_to = datetime(9999, 12, 31, tzinfo=timezone.utc)

        fill_records.append({
            "symbol": symbol,
            "rate": r["fundingRate"],
            "ts_from": ts_from,
            "ts_to": ts_to,
            "interval": INTERVAL,
        })

    return fill_records


def run(symbols: list[str], start: datetime, end: datetime) -> None:
    for symbol in symbols:
        raw = fetch_funding(symbol, start, end)
        fill_records = _build_fill_records(symbol, raw)
        n = update_funding_rates(fill_records)
        logger.info("[%s] updated %d perp candle rows with funding_rate", symbol, n)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(
        description="Fetch Binance funding rates from CDN and forward-fill onto perp candles"
    )
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start", required=True, help="Start month YYYY-MM")
    parser.add_argument("--end", default=None, help="End month YYYY-MM (default: current month)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    start = datetime.strptime(args.start + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = (
        datetime.strptime(args.end + "-01", "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else now
    )

    run(args.symbols, start, end)


if __name__ == "__main__":
    main()
