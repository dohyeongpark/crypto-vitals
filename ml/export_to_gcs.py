"""
BigQuery → feature-engineered parquet → GCS upload.

Run once (or whenever schema/features change) to prepare training data
that Colab can download without re-querying BigQuery each session.

Usage:
    python -m ml.export_to_gcs
    python -m ml.export_to_gcs --symbols BTC/USDT ETH/USDT \\
        --start 2025-06-01 --end 2026-06-01 \\
        --bucket parkdh0121-ml-data

GCS 저장 경로:
    gs://<bucket>/crypto-vitals/features/<SYMBOL>/<start>_<end>.parquet
    예) gs://parkdh0121-ml-data/crypto-vitals/features/BTC_USDT/20250601_20260601.parquet
"""

import argparse
import io
import logging
import os
import sys

from google.cloud import storage

from ml.data.loader   import load_symbol, BQ_PROJECT
from ml.data.features import build_features

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("export")

DEFAULT_BUCKET = f"{BQ_PROJECT}-ml-data"
GCS_PREFIX     = "crypto-vitals/features"
SYMBOLS_DEFAULT = ["BTC/USDT", "ETH/USDT"]


def symbol_to_path(symbol: str) -> str:
    """'BTC/USDT' → 'BTC_USDT'"""
    return symbol.replace("/", "_")


def ensure_bucket(client: storage.Client, bucket_name: str) -> storage.Bucket:
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        bucket = client.create_bucket(bucket_name, location="asia-northeast3")
        log.info("Created GCS bucket: gs://%s", bucket_name)
    return bucket


def upload_parquet(
    df,
    bucket:    storage.Bucket,
    blob_name: str,
) -> str:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    size_mb = buf.tell() / 1024 ** 2
    buf.seek(0)

    blob = bucket.blob(blob_name)
    blob.upload_from_file(buf, content_type="application/octet-stream")

    uri = f"gs://{bucket.name}/{blob_name}"
    log.info("Uploaded %d rows → %s  (%.1f MB)", len(df), uri, size_mb)
    return uri


def export_symbol(
    symbol:      str,
    start_date:  str | None,
    end_date:    str | None,
    bucket:      storage.Bucket,
) -> str:
    log.info("── %s ──", symbol)

    df = load_symbol(symbol, start_date=start_date, end_date=end_date)
    df = build_features(df)

    start_tag = (start_date or "all").replace("-", "")[:8]
    end_tag   = (end_date   or "now").replace("-", "")[:8]
    blob_name = f"{GCS_PREFIX}/{symbol_to_path(symbol)}/{start_tag}_{end_tag}.parquet"

    return upload_parquet(df, bucket, blob_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export feature parquet files to GCS")
    p.add_argument("--symbols", nargs="+", default=SYMBOLS_DEFAULT, metavar="SYM")
    p.add_argument("--start",   default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end",     default=None, metavar="YYYY-MM-DD")
    p.add_argument("--bucket",  default=DEFAULT_BUCKET)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log.info(
        "Export start  symbols=%s  start=%s  end=%s  bucket=%s",
        args.symbols, args.start, args.end, args.bucket,
    )

    client = storage.Client(project=BQ_PROJECT)
    bucket = ensure_bucket(client, args.bucket)

    for symbol in args.symbols:
        uri = export_symbol(symbol, args.start, args.end, bucket)
        log.info("Done: %s", uri)

    log.info("Export complete")


if __name__ == "__main__":
    main()
