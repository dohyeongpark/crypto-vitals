"""
Track B: Open interest collector — runs 24/7 via APScheduler.

Fetches /futures/data/openInterestHist (period=1h) every hour and UPSERTs
into open_interest table. Exceptions are logged and swallowed so the
scheduler never dies.

Run:
    python -m src.collectors.oi_scheduler        # foreground (docker service)
"""
import logging
import sys
import time
from datetime import datetime, timezone

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config import BINANCE_FAPI_BASE, HTTP_BACKOFF_BASE, HTTP_MAX_RETRIES, HTTP_TIMEOUT, SYMBOLS
from src.db import upsert_open_interest

logger = logging.getLogger(__name__)

_OI_URL = f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist"
_OI_PAGE_LIMIT = 500  # Binance max


# ── HTTP ──────────────────────────────────────────────────────────────────────

def _get_oi(params: dict) -> list[dict]:
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            resp = requests.get(_OI_URL, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            delay = HTTP_BACKOFF_BASE * (2 ** attempt)
            if attempt == HTTP_MAX_RETRIES:
                raise
            logger.warning("OI fetch retry %d/%d in %.1fs: %s", attempt, HTTP_MAX_RETRIES, delay, exc)
            time.sleep(delay)
    return []


# ── Collection ────────────────────────────────────────────────────────────────

def collect_oi(symbol: str, lookback_hours: int = 3) -> int:
    """
    Fetch last `lookback_hours` of OI data for symbol and UPSERT.
    Fetching a few hours back handles any missed ticks gracefully.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - lookback_hours * 3600 * 1000

    raw = _get_oi({
        "symbol": symbol,
        "period": "1h",
        "startTime": start_ms,
        "endTime": now_ms,
        "limit": _OI_PAGE_LIMIT,
    })

    if not raw:
        logger.warning("[%s] OI response empty", symbol)
        return 0

    rows = []
    for r in raw:
        ts = datetime.fromtimestamp(int(r["timestamp"]) / 1000, tz=timezone.utc)
        rows.append({
            "symbol": symbol,
            "timestamp": ts,
            "open_interest": r["sumOpenInterest"],
            "oi_value_usdt": r.get("sumOpenInterestValue"),
        })

    inserted = upsert_open_interest(rows)
    return inserted


# ── Scheduled job ─────────────────────────────────────────────────────────────

def _tick() -> None:
    """Runs every hour. Exceptions are caught so the scheduler survives."""
    now = datetime.now(timezone.utc)
    logger.info("OI tick started at %s", now.isoformat())
    total = 0
    for symbol in SYMBOLS:
        try:
            n = collect_oi(symbol)
            total += n
            logger.info("[%s] inserted %d OI rows", symbol, n)
        except Exception as exc:
            logger.error("[%s] OI collection failed: %s", symbol, exc, exc_info=True)

    # Log cumulative row count for monitoring
    try:
        from src.db import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM open_interest")
                total_rows = cur.fetchone()[0]
        logger.info("OI tick done. Inserted this tick: %d. Total rows in DB: %d", total, total_rows)
    except Exception as exc:
        logger.warning("Could not query total OI row count: %s", exc)


# ── Entry ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    logger.info("OI scheduler starting. Symbols=%s", SYMBOLS)

    # Run once immediately on startup to backfill any gap
    _tick()

    scheduler = BlockingScheduler(timezone="UTC")
    # 매 정시 5분 후 실행 (데이터가 준비되는 시간 확보)
    scheduler.add_job(_tick, CronTrigger(minute=5), id="oi_tick", max_instances=1)

    logger.info("Scheduler running — next tick at :05 of each hour")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("OI scheduler stopped.")


if __name__ == "__main__":
    main()
