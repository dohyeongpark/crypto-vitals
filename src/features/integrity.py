"""
Track A-3: Data integrity checks for market_data.

Detects missing hourly candles, duplicates, and prints a coverage summary.

Usage:
    python -m src.features.integrity
    python -m src.features.integrity --symbols BTCUSDT --market spot
"""
import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import pandas as pd

from src.config import INTERVAL, SYMBOLS
from src.db import get_engine

logger = logging.getLogger(__name__)


class CoverageReport(NamedTuple):
    symbol: str
    market_type: str
    start: datetime
    end: datetime
    total_expected: int
    total_actual: int
    missing_count: int
    duplicate_count: int


def _fetch_timestamps(symbol: str, market_type: str) -> pd.DataFrame:
    sql = """
        SELECT timestamp
        FROM   market_data
        WHERE  symbol      = %(symbol)s
          AND  market_type = %(market_type)s
          AND  interval    = '1h'
        ORDER  BY timestamp
    """
    return pd.read_sql_query(
        sql, get_engine(),
        params={"symbol": symbol, "market_type": market_type},
        parse_dates=["timestamp"],
    )


def check(symbol: str, market_type: str) -> CoverageReport:
    df = _fetch_timestamps(symbol, market_type)

    if df.empty:
        logger.warning("[%s/%s] No data in market_data", symbol, market_type)
        return CoverageReport(symbol, market_type, None, None, 0, 0, 0, 0)

    ts = df["timestamp"].dt.tz_localize("UTC") if df["timestamp"].dt.tz is None else df["timestamp"]
    ts = ts.sort_values().reset_index(drop=True)

    start = ts.iloc[0].to_pydatetime()
    end = ts.iloc[-1].to_pydatetime()

    # Expected: one candle every hour from start to end
    expected = pd.date_range(start=start, end=end, freq="1h", tz="UTC")
    total_expected = len(expected)
    total_actual = len(ts)

    missing = expected.difference(ts)
    missing_count = len(missing)

    # Duplicates (shouldn't exist given PK, but sanity check)
    duplicate_count = int(ts.duplicated().sum())

    if missing_count > 0:
        logger.warning("[%s/%s] %d missing candles", symbol, market_type, missing_count)
        for m in missing[:20]:
            logger.warning("  missing: %s", m)
        if missing_count > 20:
            logger.warning("  … and %d more", missing_count - 20)
    else:
        logger.info("[%s/%s] No missing candles", symbol, market_type)

    if duplicate_count > 0:
        logger.error("[%s/%s] %d duplicate timestamps detected!", symbol, market_type, duplicate_count)

    return CoverageReport(symbol, market_type, start, end, total_expected, total_actual, missing_count, duplicate_count)


def print_summary(reports: list[CoverageReport]) -> None:
    print("\n=== Market Data Coverage Summary ===")
    print(f"{'Symbol':<12} {'Type':<6} {'Start':<22} {'End':<22} {'Expected':>9} {'Actual':>7} {'Missing':>8} {'Dupes':>6}")
    print("-" * 100)
    for r in reports:
        if r.start is None:
            print(f"{r.symbol:<12} {r.market_type:<6} {'NO DATA':<22} {'':22} {'':>9} {'':>7} {'':>8} {'':>6}")
            continue
        ok = "OK" if r.missing_count == 0 else f"GAPS:{r.missing_count}"
        print(
            f"{r.symbol:<12} {r.market_type:<6}"
            f" {r.start.strftime('%Y-%m-%d %H:%M'):<22}"
            f" {r.end.strftime('%Y-%m-%d %H:%M'):<22}"
            f" {r.total_expected:>9}"
            f" {r.total_actual:>7}"
            f" {r.missing_count:>8}"
            f" {r.duplicate_count:>6}"
            f"  {ok}"
        )
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ")
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Check market_data integrity (missing candles, duplicates)")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--market", nargs="+", default=["spot", "perp"], choices=["spot", "perp"])
    args = parser.parse_args()

    reports = []
    for symbol in args.symbols:
        for market_type in args.market:
            reports.append(check(symbol, market_type))

    print_summary(reports)

    # Non-zero exit if any gaps found
    if any(r.missing_count > 0 or r.duplicate_count > 0 for r in reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
