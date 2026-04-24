# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools only in builder stage
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into an isolated prefix so we can copy just the packages
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for K8s security best-practices
RUN useradd --no-create-home --shell /bin/false appuser

WORKDIR /app

# Pull pre-built packages from builder
COPY --from=builder /install /usr/local

COPY main.py .

# Force unbuffered stdout so logs appear immediately in kubectl logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Runtime secrets are injected by K8s Secrets — never baked into the image
# ENV BINANCE_API_KEY and BINANCE_API_SECRET must be set at deploy time.

USER appuser

ENTRYPOINT ["python", "main.py"]
