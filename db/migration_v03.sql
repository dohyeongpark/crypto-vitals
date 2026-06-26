-- Migration v0.3-kalman (2026-06-26): Kalman β / OU parameters / cointegration columns
-- Run on existing DB: sudo docker compose exec timescaledb psql -U postgres -d cryptodb < migration_v03.sql
ALTER TABLE pair_features
  ADD COLUMN IF NOT EXISTS kalman_beta      NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS spread_kalman    NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS ou_kappa         NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS ou_halflife      NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS ou_mu            NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS ou_sigma_eq      NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS ou_zscore        NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS eg_pvalue        NUMERIC(10,8),
  ADD COLUMN IF NOT EXISTS johansen_trace   NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS entry_threshold  NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS exit_threshold   NUMERIC(20,8);
