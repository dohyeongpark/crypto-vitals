import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "cryptodb")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# ── Binance endpoints ─────────────────────────────────────────────────────────
# 공개 데이터만 사용 — API 키 불필요
BINANCE_API_BASE = "https://api.binance.com"
BINANCE_FAPI_BASE = "https://fapi.binance.com"
BINANCE_VISION_BASE = "https://data.binance.vision"

# ── Collection config ─────────────────────────────────────────────────────────
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "1h"

# ── Feature config ────────────────────────────────────────────────────────────
FEATURE_WINDOW_H = int(os.getenv("FEATURE_WINDOW_H", "168"))  # 7일
FEATURE_VERSION = os.getenv("FEATURE_VERSION", "v0.3-kalman")

# ── Phase 3: Triple-barrier ───────────────────────────────────────────────────
TB_ENTRY_Z    = float(os.getenv("TB_ENTRY_Z",   "2.5"))   # hparam grid search 최적값
TB_EXIT_Z     = float(os.getenv("TB_EXIT_Z",    "0.5"))
TB_STOP_Z     = float(os.getenv("TB_STOP_Z",    "3.5"))   # hparam grid search 최적값
TB_MAX_HOLD_H = int(os.getenv("TB_MAX_HOLD_H",  "12"))   # max hold 12h ≈ 6.7 half-lives
LABEL_VERSION = os.getenv("LABEL_VERSION",       "v1.1-tb")

# ── Phase 3: Meta-labeling ────────────────────────────────────────────────────
META_THRESHOLD = float(os.getenv("META_THRESHOLD", "0.5"))

# ── HTTP ──────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = 30
HTTP_MAX_RETRIES = 5
HTTP_BACKOFF_BASE = 2.0  # seconds
