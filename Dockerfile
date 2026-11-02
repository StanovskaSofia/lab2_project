# syntax=docker/dockerfile:1
# Збірка:  docker build -t kws-api:1.0.0 .
# CI:      docker build --build-arg EPOCHS=5 -t kws-api:ci .
# Стратегія B (образ без моделі): docker build --target runtime-base -t kws-api:nomodel .

# ================= Stage 1: trainer =================
FROM python:3.10-slim AS trainer
ARG CLASSES="yes no up down"
ARG EPOCHS=10
ARG SEED=42
ARG MAX_PER_CLASS=0
ARG MODEL_VERSION=1.0.0
WORKDIR /build
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# копіюємо лише код навчання: зміна api.py не запускає повторне навчання
COPY src/__init__.py src/data.py src/model.py src/evaluate.py src/train.py ./src/

# окремий шар з датасетом (~2.3 ГБ): кешується, поки не змінився код data.py
RUN python -m src.data --download --data-root /build/data

RUN python -m src.train --classes ${CLASSES} --epochs ${EPOCHS} --seed ${SEED} \
        --max-per-class ${MAX_PER_CLASS} --model-version ${MODEL_VERSION} \
        --data-root /build/data --out /build/artifacts


# ================= Stage 2: runtime-base (без моделі) =================
FROM python:3.10-slim AS runtime-base
WORKDIR /app
ENV MODEL_PATH=/app/artifacts/model.pt \
    PORT=8000 \
    LOG_LEVEL=info \
    UVICORN_WORKERS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/__init__.py src/api.py src/data.py src/model.py src/evaluate.py ./src/
COPY scripts/healthcheck.py ./scripts/healthcheck.py

RUN useradd --create-home appuser && mkdir -p /app/artifacts && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "/app/scripts/healthcheck.py"]

CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --workers ${UVICORN_WORKERS}"]


# ================= Stage 3: runtime (з «запеченою» моделлю) =================
FROM runtime-base AS runtime
ARG MODEL_VERSION=1.0.0
ARG GIT_COMMIT=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="kws-api" \
      org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# у фінальний образ потрапляють лише артефакти моделі
COPY --from=trainer --chown=appuser:appuser /build/artifacts /app/artifacts
