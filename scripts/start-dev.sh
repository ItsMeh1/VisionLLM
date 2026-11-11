#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

APP_URL="http://127.0.0.1:5173/"
ENGINE_URL="http://127.0.0.1:8765/health"
RUN_DIR="$ROOT/.dev"
mkdir -p "$RUN_DIR"

fail() {
  printf 'start: %s\n' "$*" >&2
  exit 1
}

if [ ! -f package.json ] || [ ! -d src ]; then
  fail "Run this from the VisionLLM app checkout on main. The website branch is static-only."
fi

engine_up() {
  curl -fsS --max-time 2 "$ENGINE_URL" >/dev/null 2>&1
}

app_up() {
  curl -fsS --max-time 2 "$APP_URL" >/dev/null 2>&1
}

port_busy() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

wait_for() {
  local check=$1
  for _ in $(seq 1 90); do
    if "$check"; then
      return 0
    fi
    sleep 1
  done
  return 1
}

ENGINE_PID=""
APP_PID=""
ENGINE_STARTED=0
APP_STARTED=0

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ -n "$APP_PID" ]; then kill "$APP_PID" 2>/dev/null || true; fi
  if [ -n "$ENGINE_PID" ]; then kill "$ENGINE_PID" 2>/dev/null || true; fi
  exit "$status"
}
trap cleanup EXIT INT TERM

if engine_up; then
  echo "Engine: already running; leaving it untouched."
else
  port_busy 8765 && fail "Port 8765 is in use but /health is not responding. Stop the stale process first."

  if [ ! -x engine/.venv/bin/python ]; then
    echo "Creating the Python engine environment..."
    if command -v uv >/dev/null 2>&1; then
      (cd engine && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt)
    elif command -v python3.12 >/dev/null 2>&1; then
      python3.12 -m venv engine/.venv
      engine/.venv/bin/pip install -r engine/requirements.txt
    else
      fail "Python 3.12 is required to create engine/.venv."
    fi
  elif ! engine/.venv/bin/python -c 'import fastapi, torch, torchvision' >/dev/null 2>&1; then
    echo "Installing the Python engine dependencies..."
    engine/.venv/bin/pip install -r engine/requirements.txt
  fi

  echo "Starting the engine..."
  engine/.venv/bin/python -m engine.server >>"$RUN_DIR/engine.log" 2>&1 &
  ENGINE_PID=$!
  ENGINE_STARTED=1
  if ! wait_for engine_up; then
    tail -30 "$RUN_DIR/engine.log" >&2 || true
    fail "The engine did not become ready."
  fi
fi

if app_up; then
  echo "App: already running at $APP_URL; leaving it untouched."
else
  port_busy 5173 && fail "Port 5173 is in use but the app is not responding at $APP_URL. Stop the stale process first."

  if [ ! -x node_modules/.bin/vite ]; then
    echo "Installing app dependencies..."
    npm install
  fi

  echo "Starting the app..."
  node node_modules/vite/bin/vite.js --host 127.0.0.1 --port 5173 --strictPort >>"$RUN_DIR/app.log" 2>&1 &
  APP_PID=$!
  APP_STARTED=1
  if ! wait_for app_up; then
    tail -30 "$RUN_DIR/app.log" >&2 || true
    fail "The app did not become ready."
  fi
fi

printf '\nVisionLLM is running\n'
printf '  App:    %s\n' "$APP_URL"
printf '  Engine: http://127.0.0.1:8765\n'
if [ "$ENGINE_STARTED" = 1 ]; then printf '  Engine log: %s\n' "$RUN_DIR/engine.log"; fi
if [ "$APP_STARTED" = 1 ]; then printf '  App log:    %s\n' "$RUN_DIR/app.log"; fi
printf '\n'

if [ "${VISIONLLM_NO_OPEN:-0}" != "1" ] && command -v open >/dev/null 2>&1; then
  open "$APP_URL" >/dev/null 2>&1 || true
fi

if [ "$ENGINE_STARTED" = 0 ] && [ "$APP_STARTED" = 0 ]; then
  echo "Both services were already running."
  exit 0
fi

echo "Press Ctrl-C to stop the services started by this command."
echo "Already-running services are left untouched."
wait
