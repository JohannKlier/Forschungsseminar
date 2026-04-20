#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER_DIR="$ROOT_DIR/trainer-service"
FRONTEND_DIR="$ROOT_DIR/gam-lab"

TRAINER_HOST="${TRAINER_HOST:-127.0.0.1}"
TRAINER_PORT="${TRAINER_PORT:-4001}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
TRAINER_URL="${TRAINER_URL:-http://localhost:$TRAINER_PORT}"

trainer_pid=""
frontend_pid=""

cleanup() {
  trap - EXIT INT TERM

  if [[ -z "$frontend_pid" && -z "$trainer_pid" ]]; then
    return
  fi

  echo
  echo "Stopping dev servers..."

  if [[ -n "$frontend_pid" ]] && kill -0 "$frontend_pid" 2>/dev/null; then
    kill "$frontend_pid" 2>/dev/null || true
  fi

  if [[ -n "$trainer_pid" ]] && kill -0 "$trainer_pid" 2>/dev/null; then
    kill "$trainer_pid" 2>/dev/null || true
  fi

  wait 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT TERM

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command python3
require_command npm

if [[ ! -x "$TRAINER_DIR/.venv/bin/python" ]]; then
  echo "Creating Python virtual environment..."
  python3 -m venv "$TRAINER_DIR/.venv"
fi

PYTHON="$TRAINER_DIR/.venv/bin/python"

check_port_available() {
  local host="$1"
  local port="$2"
  local label="$3"

  "$PYTHON" - "$host" "$port" "$label" <<'PY'
import socket
import sys

host, port, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    try:
        sock.bind((host, port))
    except OSError as exc:
        print(f"{label} port {port} on {host} is not available: {exc}", file=sys.stderr)
        sys.exit(1)
PY
}

if ! "$PYTHON" -m pip show uvicorn >/dev/null 2>&1; then
  echo "Installing trainer-service dependencies..."
  "$PYTHON" -m pip install -r "$TRAINER_DIR/requirements.txt"
fi

if ! compgen -G "$TRAINER_DIR/data/*" >/dev/null; then
  echo "Downloading trainer datasets..."
  (
    cd "$TRAINER_DIR"
    "$PYTHON" -m trainer_service.datasets
  )
fi

if ! compgen -G "$TRAINER_DIR/models/*.json" >/dev/null; then
  echo "Generating preset trainer models..."
  (
    cd "$TRAINER_DIR"
    "$PYTHON" -m trainer_service.generate_models
  )
fi

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm --prefix "$FRONTEND_DIR" install
fi

check_port_available "$TRAINER_HOST" "$TRAINER_PORT" "Trainer service"
check_port_available "127.0.0.1" "$FRONTEND_PORT" "Frontend"

echo "Starting trainer service at http://$TRAINER_HOST:$TRAINER_PORT"
(
  cd "$TRAINER_DIR"
  "$PYTHON" -m uvicorn trainer_service.api:app --reload --host "$TRAINER_HOST" --port "$TRAINER_PORT"
) &
trainer_pid=$!

echo "Starting frontend at http://localhost:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  TRAINER_URL="$TRAINER_URL" NEXT_PUBLIC_TRAINER_URL="$TRAINER_URL" npm run dev -- --port "$FRONTEND_PORT"
) &
frontend_pid=$!

echo
echo "Dev stack is running:"
echo "  Frontend:        http://localhost:$FRONTEND_PORT"
echo "  Trainer service: $TRAINER_URL"
echo
echo "Press Ctrl+C to stop both."

wait -n "$trainer_pid" "$frontend_pid"
