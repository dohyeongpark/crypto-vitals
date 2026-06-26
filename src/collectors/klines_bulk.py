"""
Track A-1: Bulk historical klines downloader via data.binance.vision.

data.binance.vision monthly ZIP → CSV → TimescaleDB UPSERT.
timestamp = candle CLOSE time (UTC) — no lookahead.

Usage:
    python -m src.collectors.klines_bulk --months 1          # last 1 month
    python -m src.collectors.klines_bulk --start 2023-01     # from 2023-01
    python -m src.collectors.klines_bulk --start 2023-01 --end 2023-03
    python -m src.collectors.klines_bulk --symbols BTCUSDT --market spot
"""
import argparse
import csv
import io
import logging
import sys
import time
import zipfile
from datetime import datetime, timezone
from itertools import product
from typing import Iterator

import requests

from src.config import BINANCE_VISION_BASE, HTTP_BACKOFF_BASE, HTTP_MAX_RETRIES, HTTP_TIMEOUT, INTERVAL, SYMBOLS
from src.db import upsert_market_data

logger = logging.getLogger(__name__)


# ── URL builders ──────────────────────────────────────────────────────────────

def _klines_url(symbol: str, market_type: str, year: int, month: int) -> str:
    interval = INTERVAL
    fname = f"{symbol}-{interval}-{year}-{month:02d}.zip"
    if market_type == "spot":
        return f"{BINANCE_VISION_BASE}/data/spot/monthly/klines/{symbol}/{interval}/{fname}"
    if market_type == "perp":
        return f"{BINANCE_VISION_BASE}/data/futures/um/monthly/klines/{symbol}/{interval}/{fname}"
    raise ValueError(f"Unknown market_type: {market_type}")


# ── HTTP with retry / backoff ─────────────────────────────────────────────────

def _download_zip(url: str) -> bytes | None:
    """Download ZIP. Returns None on 404 (month not published yet)."""
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT)
            if resp.status_code == 404:
                logger.debug("404 (not published): %s", url)
                return None
            resp.raise_for_status()
            return resp.content
        except requests.exceptions.RequestException as exc:
            delay = HTTP_BACKOFF_BASE * (2 ** attempt)
            if attempt == HTTP_MAX_RETRIES:
                logger.error("Failed after %d attempts: %s — %s", HTTP_MAX_RETRIES, url, exc)
                raise
            logger.warning("Retry %d/%d in %.1fs: %s", attempt, HTTP_MAX_RETRIES, delay, exc)
            time.sleep(delay)
    return None  # unreachable


# ── CSV parsing ───────────────────────────────────────────────────────────────

# Binance Vision klines CSV column order (0-indexed):
# 0: open_time_ms, 1: open, 2: high, 3: low, 4: close, 5: volume,
# 6: close_time_ms, 7: quote_volume, 8: trades_count,
# 9: taker_buy_base_vol, 10: taker_buy_quote_vol, 11: ignore

def _parse_csv(data: bytes, symbol: str, market_type: str) -> list[dict]:
    rows: list[dict] = []
    text = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8")
    reader = csv.reader(text)

    for line in reader:
        if not line or line[0].startswith("open_time"):  # skip header if any
            continue
        try:
            close_time_raw = int(line[6])
            # Binance Vision uses ms (13-digit) for older data and µs (16-digit) for 2026+.
            # Adding 1 unit and dividing gives the clean hour boundary in both cases.
            if close_time_raw > 9_999_999_999_999:  # 16-digit → microseconds
                ts = datetime.fromtimestamp((close_time_raw + 1) / 1_000_000.0, tz=timezone.utc)
            else:  # 13-digit → milliseconds
                ts = datetime.fromtimestamp((close_time_raw + 1) / 1000.0, tz=timezone.utc)

            rows.append({
                "symbol": symbol,
                "market_type": market_type,
                "timestamp": ts,
                "interval": INTERVAL,
                "open": line[1],
                "high": line[2],
                "low": line[3],
                "close": line[4],
                "volume": line[5],
                "quote_volume": line[7] or None,
                "trades_count": int(line[8]) if line[8] else None,
                "taker_buy_volume": line[9] or None,
                "source": "binance",
            })
        except (IndexError, ValueError) as exc:
            logger.warning("Skipping malformed CSV row %s: %s", line, exc)

    return rows


# ── Month range generator ─────────────────────────────────────────────────────

def _months(start: tuple[int, int], end: tuple[int, int]) -> Iterator[tuple[int, int]]:
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# ── Main entry ────────────────────────────────────────────────────────────────

def download(
    symbols: list[str],
    market_types: list[str],
    start: tuple[int, int],
    end: tuple[int, int],
    batch_size: int = 500,
) -> None:
    total_inserted = 0
    for symbol, mtype, (year, month) in product(symbols, market_types, _months(start, end)):
        url = _klines_url(symbol, mtype, year, month)
        logger.info("[%s/%s] Downloading %d-%02d …", symbol, mtype, year, month)

        try:
            content = _download_zip(url)
        except Exception:
            continue  # logged inside _download_zip; skip to next month

        if content is None:
            continue

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_name = zf.namelist()[0]
            csv_data = zf.read(csv_name)

        rows = _parse_csv(csv_data, symbol, mtype)
        if not rows:
            logger.warning("[%s/%s] %d-%02d: no rows parsed", symbol, mtype, year, month)
            continue

        # UPSERT in batches to avoid very large transactions
        inserted = 0
        for i in range(0, len(rows), batch_size):
            inserted += upsert_market_data(rows[i : i + batch_size])
        total_inserted += inserted
        logger.info("[%s/%s] %d-%02d: %d rows → %d inserted", symbol, mtype, year, month, len(rows), inserted)

    logger.info("Done. Total inserted: %d", total_inserted)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_ym(s: str) -> tuple[int, int]:
    parts = s.split("-")
    return int(parts[0]), int(parts[1])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Bulk download Binance klines from data.binance.vision")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--market", nargs="+", default=["spot", "perp"], choices=["spot", "perp"])
    parser.add_argument("--start", default=None, help="Start YYYY-MM (default: 3 years ago)")
    parser.add_argument("--end", default=None, help="End YYYY-MM (default: last complete month)")
    parser.add_argument("--months", type=int, default=None, help="Download last N months (overrides --start/--end)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    # Default end = last complete month
    if now.month == 1:
        default_end = (now.year - 1, 12)
    else:
        default_end = (now.year, now.month - 1)

    if args.months:
        end_y, end_m = default_end
        total = end_y * 12 + end_m - (args.months - 1)
        sy, sm = divmod(total - 1, 12)
        start = (sy, sm + 1)
        end = default_end
    else:
        start = _parse_ym(args.start) if args.start else (now.year - 3, now.month)
        end = _parse_ym(args.end) if args.end else default_end

    logger.info(
        "Symbols=%s Markets=%s Range=%04d-%02d → %04d-%02d",
        args.symbols, args.market, start[0], start[1], end[0], end[1],
    )
    download(args.symbols, args.market, start, end)


if __name__ == "__main__":
    main()
