"""
Historical OHLCV backfill from Binance → BigQuery.

Usage:
  python backfill.py                          # 1년치 BTC+ETH
  python backfill.py --days 180               # 180일
  python backfill.py --start 2025-01-01       # 특정 시작일
  python backfill.py --symbols BTC/USDT       # 단일 심볼
  python backfill.py --dry-run                # BQ insert 없이 출력만
"""

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import ccxt
import pandas as pd
from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logging.Formatter.converter = time.gmtime
log = logging.getLogger("backfill")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOLS_DEFAULT   = ["BTC/USDT", "ETH/USDT"]
TIMEFRAME         = "1m"
FETCH_LIMIT       = 1000          # Binance 1분봉 최대 1000개/요청
BQ_INSERT_BATCH   = 500           # BigQuery streaming insert 배치 크기
VOLATILITY_WINDOW = 5
MAX_RETRIES       = 5
RETRY_BASE_DELAY  = 2.0

BQ_PROJECT = os.environ.get("BQ_PROJECT", "parkdh0121")
BQ_DATASET = os.environ.get("BQ_DATASET", "crypto_vitals")
BQ_TABLE   = os.environ.get("BQ_TABLE",   "ohlcv")


# ---------------------------------------------------------------------------
# Exchange
# ---------------------------------------------------------------------------
def build_exchange() -> ccxt.binance:
    exchange = ccxt.binance({
        "apiKey":        os.environ.get("BINANCE_API_KEY", ""),
        "secret":        os.environ.get("BINANCE_API_SECRET", ""),
        "enableRateLimit": True,
        "options":       {"defaultType": "spot"},
    })
    log.info("Exchange ready (authenticated=%s)", bool(exchange.apiKey))
    return exchange


# ---------------------------------------------------------------------------
# BigQuery — 이미 적재된 구간 조회
# ---------------------------------------------------------------------------
def fetch_existing_range(client: bigquery.Client, symbol: str) -> tuple[datetime | None, datetime | None]:
    """BQ에서 해당 심볼의 min/max timestamp 반환."""
    query = f"""
        SELECT MIN(timestamp) AS min_ts, MAX(timestamp) AS max_ts
        FROM `{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}`
        WHERE symbol = @symbol
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("symbol", "STRING", symbol)]
    )
    row = list(client.query(query, job_config=job_config))[0]
    return row.min_ts, row.max_ts


# ---------------------------------------------------------------------------
# Fetch with retry
# ---------------------------------------------------------------------------
def fetch_page(exchange: ccxt.binance, symbol: str, since_ms: int) -> list[list]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            candles = exchange.fetch_ohlcv(
                symbol, TIMEFRAME, since=since_ms, limit=FETCH_LIMIT
            )
            return candles
        except ccxt.RateLimitExceeded as exc:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning("Rate limit, retry %d/%d in %.1fs — %s", attempt, MAX_RETRIES, delay, exc)
            time.sleep(delay)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
            delay = RETRY_BASE_DELAY * (2 ** attempt)
            log.warning("Network error, retry %d/%d in %.1fs — %s", attempt, MAX_RETRIES, delay, exc)
            time.sleep(delay)
        except ccxt.ExchangeError as exc:
            log.error("Exchange error (unrecoverable): %s", exc)
            raise
    raise RuntimeError(f"Exhausted {MAX_RETRIES} retries for {symbol} since={since_ms}")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def compute_volatility(df: pd.DataFrame, window: int = VOLATILITY_WINDOW) -> pd.DataFrame:
    log_return = (df["close"] / df["close"].shift(1)).apply(
        lambda x: x if pd.isna(x) else math.log(x)
    )
    return df.assign(volatility_5m=log_return.rolling(window).std())


def candles_to_records(symbol: str, candles: list[list]) -> list[dict]:
    columns = ["timestamp_ms", "open", "high", "low", "close", "volume"]
    df = pd.DataFrame(candles, columns=columns)
    df = compute_volatility(df)

    now_iso = datetime.now(tz=timezone.utc).isoformat()
    records = []
    for _, row in df.iterrows():
        records.append({
            "symbol":       symbol,
            "timestamp":    datetime.fromtimestamp(
                                row["timestamp_ms"] / 1000, tz=timezone.utc
                            ).isoformat(),
            "open":         float(row["open"]),
            "high":         float(row["high"]),
            "low":          float(row["low"]),
            "close":        float(row["close"]),
            "volume":       float(row["volume"]),
            "volatility_5m": None if pd.isna(row["volatility_5m"])
                              else float(row["volatility_5m"]),
            "collected_at": now_iso,
        })
    return records


# ---------------------------------------------------------------------------
# BigQuery insert (배치)
# ---------------------------------------------------------------------------
def insert_records(client: bigquery.Client, records: list[dict], dry_run: bool) -> None:
    table_ref = f"{BQ_PROJECT}.{BQ_DATASET}.{BQ_TABLE}"
    for i in range(0, len(records), BQ_INSERT_BATCH):
        batch = records[i : i + BQ_INSERT_BATCH]
        if dry_run:
            log.info("[dry-run] would insert %d rows (batch %d)", len(batch), i // BQ_INSERT_BATCH + 1)
            continue
        errors = client.insert_rows_json(table_ref, batch)
        if errors:
            log.error("BigQuery insert errors: %s", errors)
        else:
            log.debug("Inserted %d rows", len(batch))


# ---------------------------------------------------------------------------
# 심볼 1개 백필
# ---------------------------------------------------------------------------
def backfill_symbol(
    exchange:  ccxt.binance,
    client:    bigquery.Client,
    symbol:    str,
    start_dt:  datetime,
    end_dt:    datetime,
    dry_run:   bool,
) -> int:
    # 이미 적재된 구간 확인 → 중복 최소화
    existing_min, existing_max = fetch_existing_range(client, symbol)
    if existing_min and existing_max:
        log.info(
            "%s: BQ 기존 데이터 %s ~ %s",
            symbol, existing_min.isoformat(), existing_max.isoformat(),
        )
        # 기존 데이터 이전 구간만 백필 (이후 구간은 수집기가 담당)
        if start_dt >= existing_min.replace(tzinfo=timezone.utc):
            log.info("%s: start_dt가 기존 데이터 범위 안 → 기존 min 이전부터 시작", symbol)
            end_dt = min(end_dt, existing_min.replace(tzinfo=timezone.utc) - timedelta(minutes=1))
    else:
        log.info("%s: BQ에 기존 데이터 없음, 전체 구간 백필", symbol)

    if start_dt >= end_dt:
        log.info("%s: 백필 구간 없음 (start >= end)", symbol)
        return 0

    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    total_pages   = math.ceil(total_minutes / FETCH_LIMIT)
    log.info(
        "%s: %s → %s (%d분, 약 %d 페이지)",
        symbol, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"),
        total_minutes, total_pages,
    )

    since_ms     = int(start_dt.timestamp() * 1000)
    end_ms       = int(end_dt.timestamp() * 1000)
    total_rows   = 0
    page         = 0

    while since_ms < end_ms:
        candles = fetch_page(exchange, symbol, since_ms)
        if not candles:
            log.warning("%s: 빈 응답, 루프 종료 (since=%d)", symbol, since_ms)
            break

        # end_dt 초과 캔들 제거
        candles = [c for c in candles if c[0] < end_ms]
        if not candles:
            break

        records = candles_to_records(symbol, candles)
        insert_records(client, records, dry_run)

        total_rows += len(records)
        page       += 1
        since_ms    = candles[-1][0] + 60_000   # 마지막 캔들 다음 분으로 이동

        # 진행률 로그 (10페이지마다)
        if page % 10 == 0:
            pct = min((page / total_pages) * 100, 100)
            current_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc)
            log.info(
                "%s: %.1f%% (%d/%d pages) — 현재 %s, 누적 %d행",
                symbol, pct, page, total_pages,
                current_dt.strftime("%Y-%m-%d %H:%M"), total_rows,
            )

    log.info("%s: 완료 — 총 %d행 삽입", symbol, total_rows)
    return total_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binance → BigQuery 1분봉 백필")
    parser.add_argument(
        "--symbols", nargs="+", default=SYMBOLS_DEFAULT,
        metavar="SYM", help="심볼 목록 (기본: BTC/USDT ETH/USDT)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--days", type=int, default=365,
        help="오늘 기준 몇 일 전부터 백필할지 (기본: 365)",
    )
    group.add_argument(
        "--start", type=str, metavar="YYYY-MM-DD",
        help="시작 날짜 (--days와 함께 쓸 수 없음)",
    )
    parser.add_argument(
        "--end", type=str, metavar="YYYY-MM-DD",
        help="종료 날짜 (기본: 현재)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="BQ insert 없이 진행률만 출력",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    end_dt = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end
        else datetime.now(tz=timezone.utc).replace(second=0, microsecond=0)
    )
    start_dt = (
        datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.start
        else end_dt - timedelta(days=args.days)
    )

    log.info("백필 시작: %s ~ %s, 심볼=%s, dry_run=%s",
             start_dt.date(), end_dt.date(), args.symbols, args.dry_run)

    exchange = build_exchange()
    client   = bigquery.Client(project=BQ_PROJECT)

    grand_total = 0
    for symbol in args.symbols:
        rows = backfill_symbol(exchange, client, symbol, start_dt, end_dt, args.dry_run)
        grand_total += rows

    log.info("전체 완료 — 총 %d행 삽입", grand_total)


if __name__ == "__main__":
    main()
