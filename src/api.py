from __future__ import annotations

import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as AF
from fastapi import FastAPI, File, HTTPException, UploadFile
from prometheus_client import Counter, Histogram, make_asgi_app

from src.data import SAMPLE_RATE, LogMel, fix_length
from src.model import count_params, load_checkpoint

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pt"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", str(MODEL_PATH.parent / "metrics.json")))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))
MAX_SECONDS = 10

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("kws-api")

PRED_COUNT = Counter("kws_predictions_total", "Кількість передбачень за класом", ["label"])
ERR_COUNT = Counter("kws_request_errors_total", "Некоректні запити до /predict")
LATENCY = Histogram(
    "kws_inference_latency_seconds",
    "Препроцесинг + інференс",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    if n := os.getenv("TORCH_THREADS"):
        torch.set_num_threads(int(n))
    t0 = time.perf_counter()
    model, classes = load_checkpoint(str(MODEL_PATH))  # один раз при старті
    featurizer = LogMel(n_mels=model.n_mels).eval()
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    with torch.inference_mode():  # прогрів, щоб перший запит не був повільним
        model(featurizer(torch.zeros(1, SAMPLE_RATE)))
    STATE.update(
        model=model,
        classes=classes,
        featurizer=featurizer,
        metrics=metrics,
        version=os.getenv("MODEL_VERSION") or metrics.get("model_version", "unknown"),
        size_kb=round(MODEL_PATH.stat().st_size / 1024, 1),
    )
    log.info("model loaded from %s in %.2fs, classes=%s",
             MODEL_PATH, time.perf_counter() - t0, classes)
    yield
    STATE.clear()


app = FastAPI(
    title="KWS Audio Classifier",
    description="KWS API - Docker cache experiment",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.get("/health")
def health():
    return {"status": "ok" if STATE.get("model") is not None else "loading"}


@app.get("/model-info")
def model_info():
    m = STATE["metrics"]
    return {
        "architecture": m.get("architecture", type(STATE["model"]).__name__),
        "classes": STATE["classes"],
        "params": count_params(STATE["model"]),
        "model_size_kb": STATE["size_kb"],
        "accuracy": m.get("accuracy"),  # у відсотках
        "model_version": STATE["version"],
    }


def _bad_request(msg: str) -> HTTPException:
    ERR_COUNT.inc()
    return HTTPException(status_code=400, detail=msg)


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()

    if not filename.endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Expected a .wav audio file"
        )

    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is larger than {MAX_UPLOAD_BYTES} bytes"
        )

    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read audio file as WAV: {e}"
        ) from e

    if wav.size == 0 or sr <= 0 or wav.shape[0] / sr > MAX_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio is empty or longer than {MAX_SECONDS}s"
        )

    # latency: тільки препроцесинг + інференс (декодування файлу не рахується)
    t0 = time.perf_counter()
    x = torch.from_numpy(wav).mean(dim=1)  # у моно
    if sr != SAMPLE_RATE:
        x = AF.resample(x, sr, SAMPLE_RATE)
    x = fix_length(x)
    with torch.inference_mode():
        probs = torch.softmax(STATE["model"](STATE["featurizer"](x[None])), dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    classes = STATE["classes"]
    idx = int(probs.argmax())
    LATENCY.observe(latency_ms / 1000)
    PRED_COUNT.labels(classes[idx]).inc()
    ranked = sorted(zip(classes, probs.tolist()), key=lambda kv: -kv[1])
    return {
        "label": classes[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(p, 4) for c, p in ranked},
        "latency_ms": round(latency_ms, 2),
        "model_version": STATE["version"],
    }
