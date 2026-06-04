-- =============================================================================
-- crypto_vitals.ohlcv  Data Quality Checks
-- Run each section independently in BigQuery Console or bq CLI
--
-- CLI:  bq query --use_legacy_sql=false --project_id=parkdh0121 < queries/data_quality.sql
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. OVERVIEW  — row counts, date range, symbols present
-- -----------------------------------------------------------------------------
SELECT
  symbol,
  COUNT(*)                                    AS total_rows,
  MIN(timestamp)                              AS earliest,
  MAX(timestamp)                              AS latest,
  DATE_DIFF(DATE(MAX(timestamp)), DATE(MIN(timestamp)), DAY) + 1
                                              AS span_days,
  COUNT(*) / NULLIF(
    DATE_DIFF(DATE(MAX(timestamp)), DATE(MIN(timestamp)), DAY) + 1, 0
  )                                           AS avg_rows_per_day,
  -- expected: 1440 rows/day per symbol (1-minute bars)
  ROUND(COUNT(*) / NULLIF(
    DATE_DIFF(DATE(MAX(timestamp)), DATE(MIN(timestamp)), DAY) + 1, 0
  ) / 1440.0 * 100, 2)                        AS completeness_pct
FROM `parkdh0121.crypto_vitals.ohlcv`
GROUP BY symbol
ORDER BY symbol;


-- -----------------------------------------------------------------------------
-- 2. DAILY COMPLETENESS  — rows per day (flag days under 1400 rows)
-- -----------------------------------------------------------------------------
SELECT
  symbol,
  DATE(timestamp)                             AS trade_day,
  COUNT(*)                                    AS row_count,
  1440 - COUNT(*)                             AS missing_minutes,
  CASE WHEN COUNT(*) < 1400 THEN 'WARN' ELSE 'OK' END AS status
FROM `parkdh0121.crypto_vitals.ohlcv`
GROUP BY symbol, trade_day
ORDER BY symbol, trade_day;


-- -----------------------------------------------------------------------------
-- 3. GAP DETECTION  — consecutive missing 1-minute candles
--    Returns only gaps longer than 1 minute
-- -----------------------------------------------------------------------------
WITH ordered AS (
  SELECT
    symbol,
    timestamp,
    LEAD(timestamp) OVER (PARTITION BY symbol ORDER BY timestamp) AS next_ts
  FROM `parkdh0121.crypto_vitals.ohlcv`
),
gaps AS (
  SELECT
    symbol,
    timestamp                                 AS gap_start,
    next_ts                                   AS gap_end,
    TIMESTAMP_DIFF(next_ts, timestamp, MINUTE) - 1
                                              AS missing_minutes
  FROM ordered
  WHERE TIMESTAMP_DIFF(next_ts, timestamp, MINUTE) > 1
)
SELECT *
FROM gaps
ORDER BY missing_minutes DESC
LIMIT 50;


-- -----------------------------------------------------------------------------
-- 4. DUPLICATES  — same symbol + timestamp more than once
-- -----------------------------------------------------------------------------
SELECT
  symbol,
  timestamp,
  COUNT(*) AS cnt
FROM `parkdh0121.crypto_vitals.ohlcv`
GROUP BY symbol, timestamp
HAVING cnt > 1
ORDER BY cnt DESC
LIMIT 20;


-- -----------------------------------------------------------------------------
-- 5. OHLC SANITY  — high/low/open/close logical constraints
-- -----------------------------------------------------------------------------
SELECT
  symbol,
  timestamp,
  open, high, low, close,
  CASE
    WHEN high < open  THEN 'high < open'
    WHEN high < close THEN 'high < close'
    WHEN low  > open  THEN 'low > open'
    WHEN low  > close THEN 'low > close'
    WHEN high < low   THEN 'high < low'
    WHEN open <= 0    THEN 'non-positive open'
    WHEN close <= 0   THEN 'non-positive close'
    WHEN volume < 0   THEN 'negative volume'
  END AS violation
FROM `parkdh0121.crypto_vitals.ohlcv`
WHERE
     high < open
  OR high < close
  OR low  > open
  OR low  > close
  OR high < low
  OR open <= 0
  OR close <= 0
  OR volume < 0
ORDER BY symbol, timestamp;


-- -----------------------------------------------------------------------------
-- 6. PRICE OUTLIERS  — close price > 5 rolling std-devs from daily mean
-- -----------------------------------------------------------------------------
WITH daily_stats AS (
  SELECT
    symbol,
    DATE(timestamp)  AS day,
    AVG(close)       AS mean_close,
    STDDEV(close)    AS std_close
  FROM `parkdh0121.crypto_vitals.ohlcv`
  GROUP BY symbol, day
)
SELECT
  o.symbol,
  o.timestamp,
  o.close,
  d.mean_close,
  d.std_close,
  ROUND(ABS(o.close - d.mean_close) / NULLIF(d.std_close, 0), 2) AS z_score
FROM `parkdh0121.crypto_vitals.ohlcv` o
JOIN daily_stats d
  ON o.symbol = d.symbol AND DATE(o.timestamp) = d.day
WHERE ABS(o.close - d.mean_close) / NULLIF(d.std_close, 0) > 5
ORDER BY z_score DESC;


-- -----------------------------------------------------------------------------
-- 7. VOLUME OUTLIERS  — volume > 10x daily median (simple spike detector)
-- -----------------------------------------------------------------------------
WITH daily_median AS (
  SELECT
    symbol,
    DATE(timestamp)                            AS day,
    APPROX_QUANTILES(volume, 100)[OFFSET(50)]  AS median_volume
  FROM `parkdh0121.crypto_vitals.ohlcv`
  GROUP BY symbol, day
)
SELECT
  o.symbol,
  o.timestamp,
  ROUND(o.volume, 2)          AS volume,
  ROUND(d.median_volume, 2)   AS daily_median_vol,
  ROUND(o.volume / NULLIF(d.median_volume, 0), 1) AS x_median
FROM `parkdh0121.crypto_vitals.ohlcv` o
JOIN daily_median d
  ON o.symbol = d.symbol AND DATE(o.timestamp) = d.day
WHERE o.volume > d.median_volume * 10
ORDER BY x_median DESC
LIMIT 30;


-- -----------------------------------------------------------------------------
-- 8. VOLATILITY NULL RATE  — how often volatility_5m is missing
-- -----------------------------------------------------------------------------
SELECT
  symbol,
  COUNT(*)                                    AS total_rows,
  COUNTIF(volatility_5m IS NULL)              AS null_volatility,
  ROUND(COUNTIF(volatility_5m IS NULL) / COUNT(*) * 100, 2)
                                              AS null_pct,
  ROUND(AVG(volatility_5m), 8)               AS mean_volatility,
  ROUND(MAX(volatility_5m), 8)               AS max_volatility,
  ROUND(MIN(volatility_5m), 8)               AS min_volatility
FROM `parkdh0121.crypto_vitals.ohlcv`
GROUP BY symbol;


-- -----------------------------------------------------------------------------
-- 9. COLLECTION LATENCY  — collected_at vs timestamp delta (seconds)
--    High latency may indicate retry storms or infra issues
-- -----------------------------------------------------------------------------
SELECT
  symbol,
  ROUND(AVG(TIMESTAMP_DIFF(collected_at, timestamp, SECOND)), 1)  AS avg_latency_s,
  ROUND(MAX(TIMESTAMP_DIFF(collected_at, timestamp, SECOND)), 1)  AS max_latency_s,
  ROUND(APPROX_QUANTILES(
    TIMESTAMP_DIFF(collected_at, timestamp, SECOND), 100
  )[OFFSET(95)], 1)                                               AS p95_latency_s,
  COUNTIF(TIMESTAMP_DIFF(collected_at, timestamp, SECOND) > 120)  AS late_rows_gt2min
FROM `parkdh0121.crypto_vitals.ohlcv`
GROUP BY symbol;


-- -----------------------------------------------------------------------------
-- 10. SUMMARY SCORECARD  — single-row pass/fail per check per symbol
-- -----------------------------------------------------------------------------
WITH
base AS (
  SELECT symbol, COUNT(*) AS total FROM `parkdh0121.crypto_vitals.ohlcv` GROUP BY symbol
),
dupes AS (
  SELECT symbol, COUNT(*) AS dup_pairs
  FROM (
    SELECT symbol, timestamp, COUNT(*) AS c
    FROM `parkdh0121.crypto_vitals.ohlcv` GROUP BY symbol, timestamp HAVING c > 1
  ) GROUP BY symbol
),
ohlc_violations AS (
  SELECT symbol, COUNT(*) AS violations
  FROM `parkdh0121.crypto_vitals.ohlcv`
  WHERE high < open OR high < close OR low > open OR low > close
     OR high < low OR open <= 0 OR close <= 0 OR volume < 0
  GROUP BY symbol
),
null_vol AS (
  SELECT symbol,
    ROUND(COUNTIF(volatility_5m IS NULL) / COUNT(*) * 100, 2) AS null_pct
  FROM `parkdh0121.crypto_vitals.ohlcv` GROUP BY symbol
)
SELECT
  b.symbol,
  b.total                                         AS total_rows,
  COALESCE(d.dup_pairs, 0)                        AS duplicate_pairs,
  COALESCE(v.violations, 0)                       AS ohlc_violations,
  n.null_pct                                      AS volatility_null_pct,
  CASE
    WHEN COALESCE(d.dup_pairs, 0) = 0
     AND COALESCE(v.violations, 0) = 0
     AND n.null_pct < 1.0
    THEN 'PASS' ELSE 'FAIL'
  END                                             AS overall
FROM base b
LEFT JOIN dupes d USING (symbol)
LEFT JOIN ohlc_violations v USING (symbol)
LEFT JOIN null_vol n USING (symbol)
ORDER BY symbol;
