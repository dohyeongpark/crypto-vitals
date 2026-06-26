-- Migration v0.2-regime: add regime feature columns to pair_features
-- Run once on existing DB:
--   sudo docker compose exec timescaledb psql -U postgres -d cryptodb -f /migration_v02.sql
-- (copy file into container first, or pipe via stdin)

ALTER TABLE pair_features
  ADD COLUMN IF NOT EXISTS funding_rate_y  NUMERIC(12,8),
  ADD COLUMN IF NOT EXISTS funding_rate_x  NUMERIC(12,8),
  ADD COLUMN IF NOT EXISTS funding_spread  NUMERIC(12,8),
  ADD COLUMN IF NOT EXISTS taker_ratio_y   NUMERIC(10,8),
  ADD COLUMN IF NOT EXISTS taker_ratio_x   NUMERIC(10,8),
  ADD COLUMN IF NOT EXISTS basis_y         NUMERIC(20,8),
  ADD COLUMN IF NOT EXISTS basis_x         NUMERIC(20,8);
