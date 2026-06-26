"""
Track A orchestration: one-shot historical data pipeline.

Steps (in order):
  1. klines_bulk  — download OHLCV from data.binance.vision
  2. funding      — fetch and forward-fill funding rates onto perp candles
  3. integrity    — detect missing candles, abort if critical
  4. volatility   — compute realized vol
  5. pair_spread  — compute spread / z-score features

Usage:
    # Quick end-to-end test with 1 month of data
    python -m src.pipeline --months 1

    # Full 3-year run (after end-to-end is validated)
    python -m src.pipeline --start 2023-01

    # Skip already-done steps
    python -m src.pipeline --start 2023-01 --skip-klines --skip-funding
"""
import argparse
import logging
import sys
import time
from datetime import datetime, timezone

from src.collectors.klines_bulk import download as klines_download, _parse_ym, _months
from src.collectors.funding import run as funding_run
from src.features.integrity import check as integrity_check, print_summary
from src.features.volatility import compute_and_store as vol_compute
from src.features.pair_spread import compute_and_store as spread_compute
from src.config import SYMBOLS, FEATURE_WINDOW_H, FEATURE_VERSION

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(description="Track A: end-to-end historical data pipeline")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--market", nargs="+", default=["spot", "perp"], choices=["spot", "perp"])
    parser.add_argument("--start", default=None, help="Start YYYY-MM (default: 3 years ago)")
    parser.add_argument("--end", default=None, help="End YYYY-MM (default: last complete month)")
    parser.add_argument("--months", type=int, default=None, help="Last N months (overrides --start/--end)")
    parser.add_argument("--window", type=int, default=FEATURE_WINDOW_H)
    parser.add_argument("--version", default=FEATURE_VERSION)
    parser.add_argument("--skip-klines", action="store_true")
    parser.add_argument("--skip-funding", action="store_true")
    parser.add_argument("--skip-integrity", action="store_true")
    parser.add_argument("--skip-vol", action="store_true")
    parser.add_argument("--skip-spread", action="store_true")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    if now.month == 1:
        default_end = (now.year - 1, 12)
    else:
        default_end = (now.year, now.month - 1)

    if args.months:
        em, ey = default_end
        total = em + ey * 12 - (args.months - 1)
        sy, sm = divmod(total - 1, 12)
        start = (sy + 1, sm + 1)
        end = default_end
    else:
        start = _parse_ym(args.start) if args.start else (now.year - 3, now.month)
        end = _parse_ym(args.end) if args.end else default_end

    # ── Step 1: klines ────────────────────────────────────────────────────────
    if not args.skip_klines:
        logger.info("=== Step 1/5: klines download ===")
        klines_download(args.symbols, args.market, start, end)
    else:
        logger.info("=== Step 1/5: klines SKIPPED ===")

    # ── Step 2: funding rates ─────────────────────────────────────────────────
    if not args.skip_funding:
        logger.info("=== Step 2/5: funding rates ===")
        start_dt = datetime(start[0], start[1], 1, tzinfo=timezone.utc)
        end_dt = datetime(end[0], end[1], 28, tzinfo=timezone.utc)  # safe last day
        perp_symbols = [s for s in args.symbols]
        funding_run(perp_symbols, start_dt, end_dt)
    else:
        logger.info("=== Step 2/5: funding SKIPPED ===")

    # ── Step 3: integrity check ───────────────────────────────────────────────
    if not args.skip_integrity:
        logger.info("=== Step 3/5: integrity check ===")
        reports = [integrity_check(s, m) for s in args.symbols for m in args.market]
        print_summary(reports)
        n_gaps = sum(r.missing_count for r in reports)
        if n_gaps > 0:
            logger.warning("%d missing candles detected. Continuing (check logs above).", n_gaps)
    else:
        logger.info("=== Step 3/5: integrity SKIPPED ===")

    # ── Step 4: realized volatility ───────────────────────────────────────────
    if not args.skip_vol:
        logger.info("=== Step 4/5: realized volatility ===")
        for symbol in args.symbols:
            for market_type in args.market:
                vol_compute(symbol, market_type, args.window)
    else:
        logger.info("=== Step 4/5: vol SKIPPED ===")

    # ── Step 5: pair features ─────────────────────────────────────────────────
    if not args.skip_spread:
        logger.info("=== Step 5/5: pair spread / z-score ===")
        spread_compute(args.window, args.version)
    else:
        logger.info("=== Step 5/5: spread SKIPPED ===")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
