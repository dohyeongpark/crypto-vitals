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
FEATURE_VERSION = os.getenv("FEATURE_VERSION", "v0.2-regime")

# ── HTTP ──────────────────────────────────────────────────────────────────────
HTTP_TIMEOUT = 30
HTTP_MAX_RETRIES = 5
HTTP_BACKOFF_BASE = 2.0  # seconds
