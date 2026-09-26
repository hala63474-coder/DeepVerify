#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

if [ ! -x "$PY" ]; then
  echo "[setup] Creating virtual environment..."
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT/requirements.txt"
fi

PID=$(lsof -ti tcp:5000 || true)
if [ -n "$PID" ]; then
  echo "[setup] Stopping previous server on port 5000 (pid $PID)..."
  kill -9 $PID || true
  sleep 1
fi

echo "[run] Launching DeepVerify on http://127.0.0.1:5000/"
cd "$ROOT/app"
( command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:5000/ >/dev/null 2>&1 & ) || \
( command -v open    >/dev/null && open    http://127.0.0.1:5000/ >/dev/null 2>&1 & ) || true
exec "$PY" app.py
