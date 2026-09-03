# Backend API + MMC2/MMC3 inference.
# Build from the repository root:
#   docker build -f docker/backend.Dockerfile -t hemophilia-api .
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code and the model artifacts it serves.
COPY backend_api.py config.py database.py logging_config.py ./
COPY ml/ ./ml/

# Secrets are injected as environment variables at run time, never baked in.
# (The previous Dockerfile did `COPY .env .env`, which both broke the build and
#  would have embedded credentials in the image layers.)

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8000"]
