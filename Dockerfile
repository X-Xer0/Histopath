# Sevalla-Compatible Multi-Stage Dockerfile
FROM python:3.10-slim as base

# Prevent Python from writing pyc files to disc & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies for OpenCV and ReportLab
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend /app/backend
COPY frontend /app/frontend

# Set Python path
ENV PYTHONPATH=/app

# Sevalla dynamically assigns PORT at runtime (default to 8000)
ENV PORT=8000
EXPOSE 8000

# Entrypoint command
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT}"]
