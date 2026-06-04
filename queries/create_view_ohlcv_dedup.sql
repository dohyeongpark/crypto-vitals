CREATE OR REPLACE VIEW `parkdh0121.crypto_vitals.ohlcv_dedup` AS
SELECT
  symbol,
  timestamp,
  open,
  high,
  low,
  close,
  volume,
  volatility_5m,
  collected_at
FROM `parkdh0121.crypto_vitals.ohlcv`
QUALIFY
  ROW_NUMBER() OVER (
    PARTITION BY symbol, timestamp
    ORDER BY collected_at DESC
  ) = 1;
