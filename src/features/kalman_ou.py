"""
Track A-7: Kalman filter dynamic β, OU parameter estimation, and cointegration tests.

Reads spot closes from market_data, computes:
  - kalman_beta   : time-varying hedge ratio via 1D Kalman filter (pure numpy)
  - spread_kalman : log_y - kalman_beta * log_x
  - ou_kappa      : OU mean-reversion speed κ = -ln(φ̂) [per hour]
  - ou_halflife   : ln(2) / κ [hours]
  - ou_mu         : OU equilibrium mean (rolling AR(1) intercept)
  - ou_sigma_eq   : OU equilibrium σ = σ_ε / sqrt(1 - φ̂²)
  - ou_zscore     : (spread_kalman - ou_mu) / ou_sigma_eq
  - eg_pvalue     : rolling Engle-Granger cointegration p-value
  - johansen_trace: rolling Johansen trace statistic (r=0)
  - entry_threshold / exit_threshold: 2.0 / 0.5 (Avellaneda-Lee defaults)

UPDATEs existing pair_features rows for the given feature_version.
Must run AFTER pair_spread (Step 5) and regime (Step 6).

Usage:
    python -m src.features.kalman_ou
    python -m src.features.kalman_ou --version v0.3-kalman --kalman-Q 1e-6 --kalman-R 1e-3
"""
import argparse
import logging
import time

import numpy as np
import pandas as pd

from src.config import FEATURE_VERSION, FEATURE_WINDOW_H, INTERVAL
from src.db import get_engine, update_kalman_ou_features

logger = logging.getLogger(__name__)

PAIR_ID = "ETHUSDT_BTCUSDT"
Y_SYMBOL = "ETHUSDT"
X_SYMBOL = "BTCUSDT"

# Avellaneda-Lee (2010) default entry/exit z-score thresholds
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.5


KALMAN_OVERLAP_H = 720  # warm-up bars for incremental Kalman state convergence


def _fetch_max_kalman_ts(version: str) -> pd.Timestamp | None:
    """Return latest timestamp in pair_features that already has Kalman features set."""
    sql = """
        SELECT MAX(timestamp) AS max_ts
        FROM   pair_features
        WHERE  pair_id         = 'ETHUSDT_BTCUSDT'
          AND  feature_version = %(fv)s
          AND  ou_zscore       IS NOT NULL
    """
    df = pd.read_sql_query(sql, get_engine(), params={"fv": version})
    val = df["max_ts"].iloc[0]
    if val is None or pd.isnull(val):
        return None
    ts = pd.Timestamp(val)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts


def _fetch_spot_closes(since: pd.Timestamp | None = None) -> pd.DataFrame:
    """Load spot close prices for BTC and ETH, aligned by timestamp."""
    if since is not None:
        sql = """
            SELECT timestamp, symbol, close
            FROM   market_data
            WHERE  symbol      IN ('BTCUSDT', 'ETHUSDT')
              AND  market_type = 'spot'
              AND  interval    = '1h'
              AND  timestamp   >= %(since)s
            ORDER  BY timestamp
        """
        df = pd.read_sql_query(sql, get_engine(), params={"since": since},
                               parse_dates=["timestamp"])
    else:
        sql = """
            SELECT timestamp, symbol, close
            FROM   market_data
            WHERE  symbol      IN ('BTCUSDT', 'ETHUSDT')
              AND  market_type = 'spot'
              AND  interval    = '1h'
            ORDER  BY timestamp
        """
        df = pd.read_sql_query(sql, get_engine(), parse_dates=["timestamp"])
    wide = df.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    wide = wide.dropna()
    wide.index = wide.index.tz_localize("UTC") if wide.index.tz is None else wide.index
    return wide


def _kalman_filter(
    log_y: np.ndarray,
    log_x: np.ndarray,
    Q: float = 1e-6,
    R: float = 1e-3,
) -> np.ndarray:
    """
    Online 1D Kalman filter for dynamic hedge ratio β.

    State-space model:
      β_t = β_{t-1} + w_t,   w ~ N(0, Q)   [β drifts as random walk]
      y_t = β_t * x_t + v_t, v ~ N(0, R)   [observation]

    Q controls how fast β can change. R is the observation noise variance.
    Both are tunable via CLI flags for Phase 4 optimization.
    """
    n = len(log_y)
    betas = np.full(n, np.nan)

    # Initialize: β_0 = 1.0 (reasonable prior for ETH/BTC log-price ratio)
    # P_0 scaled so first Kalman gain ≈ 0.045
    beta_est = 1.0
    P_est = R / (log_x[0] ** 2) if log_x[0] != 0 else 1e-5

    for i in range(n):
        x_i = log_x[i]
        y_i = log_y[i]

        # Predict
        P_pred = P_est + Q

        # Update
        S = x_i * x_i * P_pred + R
        if S == 0:
            betas[i] = beta_est
            continue
        K = P_pred * x_i / S
        innov = y_i - beta_est * x_i
        beta_est = beta_est + K * innov
        P_est = (1.0 - K * x_i) * P_pred

        betas[i] = beta_est

    return betas


def _rolling_ar1_ou(spread: np.ndarray, window: int) -> dict:
    """
    Rolling AR(1) OLS on spread_kalman to estimate OU parameters.

    AR(1): s_t = α + φ * s_{t-1} + ε_t
    OU mapping:
      κ        = -ln(φ)              [mean-reversion speed, per hour]
      half-life = ln(2) / κ          [hours]
      μ        = α / (1 - φ)         [equilibrium mean]
      σ_eq     = σ_ε / sqrt(1 - φ²) [equilibrium σ, discrete-time formula]

    Non-stationary windows (φ ≥ 1 or φ ≤ 0): OU outputs = NaN, ou_mu still valid.
    """
    n = len(spread)
    ou_kappa = np.full(n, np.nan)
    ou_halflife = np.full(n, np.nan)
    ou_mu = np.full(n, np.nan)
    ou_sigma_eq = np.full(n, np.nan)

    for i in range(window, n):
        s = spread[i - window : i]
        y = s[1:]
        x = s[:-1]

        var_x = np.var(x, ddof=1)
        if var_x == 0:
            continue

        phi = np.cov(x, y, ddof=1)[0, 1] / var_x
        alpha = np.mean(y) - phi * np.mean(x)

        # ou_mu is always valid (rolling regression intercept / (1-phi))
        if abs(1 - phi) > 1e-10:
            mu = alpha / (1 - phi)
        else:
            mu = np.nan
        ou_mu[i] = mu

        # OU parameters only valid when spread is stationary
        if phi <= 0.0 or phi >= 1.0:
            continue

        kappa = -np.log(phi)
        halflife = np.log(2) / kappa
        ou_kappa[i] = kappa
        ou_halflife[i] = halflife

        residuals = y - alpha - phi * x
        sigma_eps = np.std(residuals, ddof=2) if len(residuals) > 2 else np.nan
        if sigma_eps is not None and not np.isnan(sigma_eps):
            denom = 1.0 - phi ** 2
            ou_sigma_eq[i] = sigma_eps / np.sqrt(denom) if denom > 0 else np.nan

    return {
        "ou_kappa": ou_kappa,
        "ou_halflife": ou_halflife,
        "ou_mu": ou_mu,
        "ou_sigma_eq": ou_sigma_eq,
    }


def _rolling_eg(log_y: np.ndarray, log_x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Engle-Granger cointegration test.
    p-value < 0.05 → reject null of no cointegration.
    """
    from statsmodels.tsa.stattools import coint

    n = len(log_y)
    eg_pvalue = np.full(n, np.nan)

    for i in range(window, n):
        y_w = log_y[i - window : i]
        x_w = log_x[i - window : i]
        try:
            _, pval, _ = coint(y_w, x_w, trend="c")
            eg_pvalue[i] = pval
        except Exception:
            pass  # leave NaN on degenerate window

    return eg_pvalue


def _rolling_johansen(log_y: np.ndarray, log_x: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling Johansen trace test (r=0: null of no cointegrating vectors).
    Trace stat > ~15.41 → reject null at 95% for 2 variables.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    n = len(log_y)
    johansen_trace = np.full(n, np.nan)

    for i in range(window, n):
        y_w = log_y[i - window : i]
        x_w = log_x[i - window : i]
        try:
            data_w = np.column_stack([y_w, x_w])
            jh = coint_johansen(data_w, det_order=0, k_ar_diff=1)
            johansen_trace[i] = jh.lr1[0]
        except Exception:
            pass  # leave NaN on degenerate window

    return johansen_trace


def compute_and_store(
    window: int,
    version: str,
    Q: float = 1e-6,
    R: float = 1e-3,
    incremental: bool = False,
) -> int:
    """
    Compute all Phase 2 features and UPDATE pair_features rows for `version`.
    Returns number of rows updated.
    """
    if incremental:
        max_ts = _fetch_max_kalman_ts(version)
        # Fetch KALMAN_OVERLAP_H bars before max_ts so the Kalman filter
        # converges to a good state estimate before hitting the new rows.
        # The UPDATE for overlap rows is harmless — it re-computes with warm state.
        fetch_since = (max_ts - pd.Timedelta(hours=KALMAN_OVERLAP_H)) if max_ts else None
        logger.info(
            "Incremental mode: fetching from %s (max_processed=%s, overlap=%dh)",
            fetch_since, max_ts, KALMAN_OVERLAP_H,
        )
    else:
        fetch_since = None

    wide = _fetch_spot_closes(since=fetch_since)
    if wide.empty or Y_SYMBOL not in wide.columns or X_SYMBOL not in wide.columns:
        logger.error("Missing spot close data for %s or %s", Y_SYMBOL, X_SYMBOL)
        return 0

    log_y = np.log(wide[Y_SYMBOL].astype(float).values)
    log_x = np.log(wide[X_SYMBOL].astype(float).values)
    timestamps = wide.index

    logger.info("Running Kalman filter (N=%d, Q=%.0e, R=%.0e)...", len(log_y), Q, R)
    kalman_beta = _kalman_filter(log_y, log_x, Q=Q, R=R)
    spread_kalman = log_y - kalman_beta * log_x

    logger.info("Estimating OU parameters (window=%d)...", window)
    ou = _rolling_ar1_ou(spread_kalman, window)
    ou_kappa = ou["ou_kappa"]
    ou_halflife = ou["ou_halflife"]
    ou_mu = ou["ou_mu"]
    ou_sigma_eq = ou["ou_sigma_eq"]

    # ou_zscore: NaN where ou_sigma_eq is 0 or NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        ou_zscore = np.where(
            (ou_sigma_eq > 0) & ~np.isnan(ou_sigma_eq),
            (spread_kalman - ou_mu) / ou_sigma_eq,
            np.nan,
        )

    logger.info("Running rolling EG test (window=%d, N=%d windows)...", window, len(log_y) - window)
    eg_pvalue = _rolling_eg(log_y, log_x, window)

    logger.info("Running rolling Johansen test (window=%d)...", window)
    johansen_trace = _rolling_johansen(log_y, log_x, window)

    def _safe(val: float) -> float | None:
        if val is None:
            return None
        try:
            f = float(val)
            return None if np.isnan(f) else f
        except (TypeError, ValueError):
            return None

    rows = []
    for i, ts in enumerate(timestamps):
        # Skip burn-in: rows without OU params (NaN kalman_beta or ou from first window)
        if np.isnan(kalman_beta[i]):
            continue
        rows.append({
            "timestamp": ts.to_pydatetime(),
            "kalman_beta": _safe(kalman_beta[i]),
            "spread_kalman": _safe(spread_kalman[i]),
            "ou_kappa": _safe(ou_kappa[i]),
            "ou_halflife": _safe(ou_halflife[i]),
            "ou_mu": _safe(ou_mu[i]),
            "ou_sigma_eq": _safe(ou_sigma_eq[i]),
            "ou_zscore": _safe(ou_zscore[i]),
            "eg_pvalue": _safe(eg_pvalue[i]),
            "johansen_trace": _safe(johansen_trace[i]),
            "entry_threshold": ENTRY_THRESHOLD,
            "exit_threshold": EXIT_THRESHOLD,
        })

    if not rows:
        logger.warning("No valid rows to update (window=%d, version=%s)", window, version)
        return 0

    n = update_kalman_ou_features(rows, version)
    logger.info(
        "Updated %d pair_features rows with Kalman/OU features (version=%s)",
        n, version,
    )
    return n


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime

    parser = argparse.ArgumentParser(
        description="Compute Kalman β / OU params / cointegration and update pair_features"
    )
    parser.add_argument("--window", type=int, default=FEATURE_WINDOW_H)
    parser.add_argument("--version", default=FEATURE_VERSION)
    parser.add_argument("--kalman-Q", type=float, default=1e-6, dest="kalman_Q",
                        help="Kalman process noise variance (controls β adaptation speed)")
    parser.add_argument("--kalman-R", type=float, default=1e-3, dest="kalman_R",
                        help="Kalman observation noise variance (log-price residual variance)")
    parser.add_argument("--incremental", action="store_true")
    args = parser.parse_args()

    compute_and_store(args.window, args.version, Q=args.kalman_Q, R=args.kalman_R,
                      incremental=args.incremental)


if __name__ == "__main__":
    main()
