"""Вимірювання latency /predict: server (latency_ms з відповіді) і client (end-to-end).

python scripts/benchmark.py --file samples/yes.wav --n 200 --warmup 10 [--max-p95-ms 50]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import requests


def stats(v):
    a = np.array(v)
    return {
        "mean": round(float(a.mean()), 2),
        "p50": round(float(np.percentile(a, 50)), 2),
        "p95": round(float(np.percentile(a, 95)), 2),
    }


def run_benchmark(url: str, file: str, n: int = 100, warmup: int = 10) -> dict:
    with open(file, "rb") as f:
        audio = f.read()
    s = requests.Session()

    def call():
        t = time.perf_counter()
        r = s.post(f"{url}/predict", files={"file": ("a.wav", audio, "audio/wav")}, timeout=30)
        r.raise_for_status()
        return (time.perf_counter() - t) * 1000, r.json()["latency_ms"]

    for _ in range(warmup):
        call()
    client, server = zip(*(call() for _ in range(n)))
    return {"n": n, "server_ms": stats(server), "client_ms": stats(client)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--file", default="samples/yes.wav")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max-p95-ms", type=float, default=None, help="quality gate для CI")
    a = ap.parse_args()
    res = run_benchmark(a.url, a.file, a.n, a.warmup)
    print(json.dumps(res, indent=2))
    if a.max_p95_ms is not None and res["client_ms"]["p95"] > a.max_p95_ms:
        print(f"FAIL: client p95 {res['client_ms']['p95']} > {a.max_p95_ms} ms")
        sys.exit(1)
