-- schema.sql — TimescaleDB schema for BTC/ETH stat arb data pipeline
-- 원시 데이터는 절대 수정하지 않음. 파생 피처는 별도 테이블.
-- 모든 timestamp = 봉 마감(close) 시각 UTC.

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 1: market_data (원시 OHLCV, 불변)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_data (
    symbol           VARCHAR(20)    NOT NULL,          -- 'BTCUSDT', 'ETHUSDT'
    market_type      VARCHAR(10)    NOT NULL,          -- 'spot' | 'perp'
    timestamp        TIMESTAMPTZ    NOT NULL,          -- 봉 close time (UTC)
    interval         VARCHAR(5)     NOT NULL,          -- '1h'
    open             NUMERIC(20,8)  NOT NULL,
    high             NUMERIC(20,8)  NOT NULL,
    low              NUMERIC(20,8)  NOT NULL,
    close            NUMERIC(20,8)  NOT NULL,
    volume           NUMERIC(30,8)  NOT NULL,
    quote_volume     NUMERIC(30,8),
    trades_count     INTEGER,
    taker_buy_volume NUMERIC(30,8),                   -- taker buy base asset volume
    funding_rate     NUMERIC(12,8),                   -- perp 전용, 8h → 1h forward-fill
    realized_vol     NUMERIC(20,8),                   -- 수집 후 계산 (A-4)
    collected_at     TIMESTAMPTZ    NOT NULL DEFAULT now(),
    source           VARCHAR(20)    NOT NULL DEFAULT 'binance',
    PRIMARY KEY (symbol, market_type, interval, timestamp)
);

SELECT create_hypertable(
    'market_data', 'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

CREATE INDEX IF NOT EXISTS market_data_symbol_type_ts
    ON market_data (symbol, market_type, timestamp DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 2: open_interest (트랙 B, 매시간 적재)
-- OI는 소급 불가(Binance 최근 1개월만 제공)이므로 market_data와 분리.
-- 현재 미사용: us-central1 IP에서 fapi.binance.com HTTP 451 영구 차단.
-- 스키마 보존 — 향후 ablation 또는 프록시/리전 전환 시 재활성 가능.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS open_interest (
    symbol         VARCHAR(20)    NOT NULL,
    timestamp      TIMESTAMPTZ    NOT NULL,            -- 정시 정렬 (UTC)
    open_interest  NUMERIC(30,8)  NOT NULL,
    oi_value_usdt  NUMERIC(30,8),
    collected_at   TIMESTAMPTZ    NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, timestamp)
);

SELECT create_hypertable(
    'open_interest', 'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

CREATE INDEX IF NOT EXISTS oi_symbol_ts
    ON open_interest (symbol, timestamp DESC);

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 3: pair_features (파생 — spread / zscore)
-- feature_version 컬럼으로 재현성 보장.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pair_features (
    pair_id          VARCHAR(30)    NOT NULL,          -- 'ETHUSDT_BTCUSDT'
    interval         VARCHAR(5)     NOT NULL,
    timestamp        TIMESTAMPTZ    NOT NULL,          -- 봉 close time (UTC)
    log_price_y      NUMERIC(20,8),                   -- log(ETH close)
    log_price_x      NUMERIC(20,8),                   -- log(BTC close)
    hedge_beta       NUMERIC(20,8),                   -- rolling OLS β (Phase1)
    spread           NUMERIC(20,8),                   -- log_y - β * log_x
    spread_mean      NUMERIC(20,8),                   -- rolling mean of spread
    spread_std       NUMERIC(20,8),                   -- rolling std of spread
    zscore           NUMERIC(20,8),                   -- robust z-score (MAD)
    -- v0.2-regime: regime feature columns (basis / taker / funding)
    funding_rate_y   NUMERIC(12,8),                   -- ETH perp funding rate (forward-filled)
    funding_rate_x   NUMERIC(12,8),                   -- BTC perp funding rate (forward-filled)
    funding_spread   NUMERIC(12,8),                   -- funding_rate_y - funding_rate_x
    taker_ratio_y    NUMERIC(10,8),                   -- ETH taker_buy_volume / volume
    taker_ratio_x    NUMERIC(10,8),                   -- BTC taker_buy_volume / volume
    basis_y          NUMERIC(20,8),                   -- ETH (perp_close - spot_close) / spot_close
    basis_x          NUMERIC(20,8),                   -- BTC (perp_close - spot_close) / spot_close
    feature_version  VARCHAR(20)    NOT NULL,          -- 'v0.2-regime'
    computed_at      TIMESTAMPTZ    NOT NULL DEFAULT now(),
    PRIMARY KEY (pair_id, interval, timestamp, feature_version)
);

SELECT create_hypertable(
    'pair_features', 'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

CREATE INDEX IF NOT EXISTS pair_features_pair_ts
    ON pair_features (pair_id, interval, timestamp DESC);
