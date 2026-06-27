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
    -- v0.3-kalman: Kalman filter β / OU parameters / cointegration
    kalman_beta      NUMERIC(20,8),                   -- Kalman filter dynamic hedge ratio β
    spread_kalman    NUMERIC(20,8),                   -- log_y - kalman_beta * log_x
    ou_kappa         NUMERIC(20,8),                   -- OU mean-reversion speed κ (per hour)
    ou_halflife      NUMERIC(20,8),                   -- ln(2)/κ in hours
    ou_mu            NUMERIC(20,8),                   -- OU long-run equilibrium mean
    ou_sigma_eq      NUMERIC(20,8),                   -- OU equilibrium σ = σ_ε / sqrt(1-φ²)
    ou_zscore        NUMERIC(20,8),                   -- (spread_kalman - ou_mu) / ou_sigma_eq
    eg_pvalue        NUMERIC(10,8),                   -- rolling Engle-Granger cointegration p-value
    johansen_trace   NUMERIC(20,8),                   -- rolling Johansen trace stat (r=0)
    entry_threshold  NUMERIC(20,8),                   -- entry z-score threshold (2.0)
    exit_threshold   NUMERIC(20,8),                   -- exit z-score threshold (0.5)
    feature_version  VARCHAR(20)    NOT NULL,          -- 'v0.3-kalman'
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

-- ─────────────────────────────────────────────────────────────────────────────
-- TABLE 4: ml_labels (Phase 3 — triple-barrier + meta-labeling)
-- 라벨은 미래 데이터를 사용(설계상)하므로 pair_features와 물리적으로 분리.
-- entry_timestamp = pair_features.timestamp (진입 바의 close 시각).
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml_labels (
    pair_id         VARCHAR(30)   NOT NULL,
    interval        VARCHAR(5)    NOT NULL,
    entry_timestamp TIMESTAMPTZ   NOT NULL,  -- 진입 바 close 시각 (pair_features.timestamp 연결)
    feature_version VARCHAR(20)   NOT NULL,  -- pair_features.feature_version 연결 키
    label_version   VARCHAR(20)   NOT NULL,  -- e.g. 'v1.0-tb'

    -- 진입 정보
    entry_side      SMALLINT      NOT NULL,  -- +1=long spread, -1=short spread
    entry_zscore    NUMERIC(20,8),

    -- Triple-barrier 결과
    exit_timestamp  TIMESTAMPTZ,
    hold_bars       SMALLINT,
    tb_label        SMALLINT      NOT NULL,  -- +1=exit hit, -1=stop hit, 0=timeout

    -- 수익 지표
    delta_zscore    NUMERIC(20,8),           -- |entry_z| - |exit_z|; 양수 = 평균회귀
    spread_return   NUMERIC(20,8),           -- signed % spread_kalman 변화

    -- TB 파라미터 (재현성 보장)
    stop_z          NUMERIC(10,4) NOT NULL DEFAULT 3.0,
    max_hold_h      SMALLINT      NOT NULL DEFAULT 12,

    -- 메타레이블 (훈련 후 채워짐)
    meta_prob       NUMERIC(10,8),           -- LightGBM P(tb_label = +1)
    meta_label      SMALLINT,                -- 1 if meta_prob > META_THRESHOLD else 0

    computed_at     TIMESTAMPTZ   NOT NULL DEFAULT now(),
    PRIMARY KEY (pair_id, interval, entry_timestamp, feature_version, label_version)
);

SELECT create_hypertable(
    'ml_labels', 'entry_timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'
);

CREATE INDEX IF NOT EXISTS ml_labels_pair_ts
    ON ml_labels (pair_id, interval, entry_timestamp DESC);

CREATE INDEX IF NOT EXISTS ml_labels_meta_null
    ON ml_labels (pair_id, feature_version, label_version)
    WHERE meta_prob IS NULL;
