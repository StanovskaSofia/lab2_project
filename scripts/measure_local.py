"""Локальний запуск (без Docker): холодний старт, RAM, latency, розмір venv.

Запускайте з активованого venv:  python scripts/measure_local.py --venv .venv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import psutil
import requests
from benchmark import run_benchmark


def dir_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total / 1024 / 1024


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=".venv")
    ap.add_argument("--file", default="samples/yes.wav")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()

    env = {**os.environ, "MODEL_PATH": "artifacts/model.pt"}
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--port", str(a.port)], env=env
    )
    url = f"http://localhost:{a.port}"
    try:
        while True:
            try:
                if requests.get(f"{url}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(0.05)
        cold = time.perf_counter() - t0
        res = run_benchmark(url, a.file, a.n)
        rss = psutil.Process(proc.pid).memory_info().rss / 1024 / 1024
        model_mb = os.path.getsize("artifacts/model.pt") / 1024 / 1024
        print(f"cold_start_s={cold:.2f}")
        print(f"ram_mb={rss:.0f}")
        print(f"env_size_mb={dir_size_mb(a.venv) + model_mb:.0f}  (venv + модель)")
        print(res)
    finally:
        proc.terminate()
