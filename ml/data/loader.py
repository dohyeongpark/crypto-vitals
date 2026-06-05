"""
BigQuery → pandas DataFrame loader.
Reads from the ohlcv_dedup view so duplicates are pre-excluded.
"""

import os
import logging
import pandas as pd
from google.cloud import bigquery

log = logging.getLogger(__name__)

BQ_PROJECT = os.environ.get("BQ_PROJECT", "parkdh0121")
BQ_DATASET = os.environ.get("BQ_DATASET", "crypto_vitals")
BQ_VIEW    = "ohlcv_dedup"

OUTPUT_COLS = ["timestamp", "open", "high", "low", "close", "volume", "volatility_5m"]


def load_symbol(
    symbol:     str,
    start_date: str | None = None,
    end_date:   str | None = None,
    project:    str = BQ_PROJECT,
) -> pd.DataFrame:
    """
    Load 1-minute OHLCV data for a single symbol from BigQuery.

    Args:
        symbol:     e.g. "BTC/USDT"
        start_date: inclusive lower bound, e.g. "2025-06-01" (optional)
        end_date:   exclusive upper bound, e.g. "2026-06-01" (optional)
        project:    GCP project ID

    Returns:
        DataFrame sorted by timestamp (UTC-aware) with columns:
        [timestamp, open, high, low, close, volume, volatility_5m]
    """
    client = bigquery.Client(project=project)

    conditions = ["symbol = @symbol"]
    params: list[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("symbol", "STRING", symbol)
    ]

    if start_date:
        conditions.append("timestamp >= @start_date")
        params.append(bigquery.ScalarQueryParameter("start_date", "TIMESTAMP", start_date))
    if end_date:
        conditions.append("timestamp < @end_date")
        params.append(bigquery.ScalarQueryParameter("end_date", "TIMESTAMP", end_date))

    where = " AND ".join(conditions)
    query = f"""
        SELECT {", ".join(OUTPUT_COLS)}
        FROM `{project}.{BQ_DATASET}.{BQ_VIEW}`
        WHERE {where}
        ORDER BY timestamp
    """

    log.info("Querying BQ: symbol=%s start=%s end=%s", symbol, start_date, end_date)
    df = (
        client.query(query, job_config=bigquery.QueryJobConfig(query_parameters=params))
        .to_dataframe()
    )
    df = df.assign(timestamp=pd.to_datetime(df["timestamp"], utc=True))

    log.info(
        "Loaded %d rows  [%s → %s]",
        len(df),
        df["timestamp"].min().isoformat(),
        df["timestamp"].max().isoformat(),
    )
    return df.reset_index(drop=True)
