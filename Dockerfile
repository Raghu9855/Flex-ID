# ==============================================================================
# Multi-stage Production Dockerfile for FLEX-ID Backend
# ==============================================================================

# Stage 1: Build stage with compiler toolchain and virtual environment
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to isolate installed dependencies cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 2: Minimal runtime image
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH="/app" \
    PORT=8000

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root system user
RUN groupadd -g 1001 appgroup && \
    useradd -u 1001 -g appgroup -m -s /bin/bash appuser

# Copy application and ML modules
COPY --chown=appuser:appgroup app/ /app/app/
COPY --chown=appuser:appgroup ml/ /app/ml/
COPY --chown=appuser:appgroup data/ /app/data/
COPY --chown=appuser:appgroup results/ /app/results/
COPY --chown=appuser:appgroup .env.example /app/.env.example

# Grant appuser ownership of the /app working directory so SQLite can create and write flex_id.db
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose standard local port (8000) and Render default port (10000)
EXPOSE 8000 10000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Production invocation: 1 worker to ensure memory strictly remains under 512MB RAM free tier
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
