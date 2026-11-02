#!/usr/bin/env bash
# Linux / WSL / Git-Bash. Використання: scripts/measure_docker.sh kws-api:1.0.0
set -euo pipefail
IMAGE="${1:-kws-api:1.0.0}"; NAME=kws-bench; PORT=8000
docker rm -f "$NAME" >/dev/null 2>&1 || true

START=$(date +%s.%N)
docker run -d --name "$NAME" -p $PORT:8000 "$IMAGE" >/dev/null
until curl -sf "http://localhost:$PORT/health" >/dev/null; do sleep 0.05; done
END=$(date +%s.%N)
python3 -c "print('cold_start_s=%.2f' % ($END - $START))"

python3 scripts/benchmark.py --url "http://localhost:$PORT" --n 200
echo -n "ram: "; docker stats --no-stream --format '{{.MemUsage}}' "$NAME"
echo -n "image_size: "; docker images "$IMAGE" --format '{{.Size}}'
echo "--- accuracy у контейнері (потрібен інтернет для датасету) ---"
docker exec "$NAME" python -m src.evaluate --data-root /tmp/data
docker rm -f "$NAME" >/dev/null
