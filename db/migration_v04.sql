-- migration_v04.sql — Phase 3: ml_labels table for triple-barrier labels + meta-labeling
-- 라벨은 미래 데이터를 사용(설계상)하므로 pair_features와 물리적으로 분리.
-- entry_timestamp = pair_features.timestamp (진입 바의 close 시각).

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

-- Partial index for finding un-scored rows efficiently
CREATE INDEX IF NOT EXISTS ml_labels_meta_null
    ON ml_labels (pair_id, feature_version, label_version)
    WHERE meta_prob IS NULL;
