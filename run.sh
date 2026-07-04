#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env and fill in your NVIDIA_API_KEYS"
    exit 1
fi

exec uvicorn main:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8000}"