# =============================================================================
#  HistologyAI - nucleus segmentation & morphometry
#  Multi-stage build for container hosts that assign a $PORT at runtime
#  (Sevalla, Render, Railway, Fly, plain docker run).
# =============================================================================

# ---------------------------------------------------------------- builder ----
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# build-essential is only needed to compile any wheel that has no prebuilt
# manylinux build; it stays in this stage and never reaches the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------- runtime ----
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000

WORKDIR /app

# libgl1 + libglib2.0-0 are the OpenCV runtime dependencies.
# curl is deliberately absent; the healthcheck uses Python's standard library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

COPY backend /app/backend
COPY frontend /app/frontend

# run unprivileged
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Inference is CPU-bound and can pin a fractional CPU for a minute at a time, so
# the probe needs room: on an S1 pod (0.5 CPU) a saturated core can delay even a
# trivial endpoint, and four consecutive failures would mark the container
# unhealthy while a job is still running.
HEALTHCHECK --interval=30s --timeout=20s --start-period=60s --retries=5 \
    CMD python -c "import os,urllib.request,sys; \
url=f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/api/health\"; \
sys.exit(0 if urllib.request.urlopen(url, timeout=6).status == 200 else 1)"

# A single worker: the inference is CPU-bound, so extra processes would only
# compete for the same cores. CPU work is dispatched to a thread pool by the
# application, which keeps the event loop free to answer /api/health.
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
