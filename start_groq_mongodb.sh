#!/usr/bin/env bash
set -euo pipefail

# Optional: ensure Python path is set to current directory
export PYTHONPATH="${PYTHONPATH:-.}"

# Default host/port for Render; PORT is injected by Render
HOST="0.0.0.0"
PORT_TO_USE="${PORT:-8000}"

# Start FastAPI via Uvicorn using the app exported in app.py (which re-exports from app_groq_ultimate.py)
exec python -m uvicorn app:app --host "$HOST" --port "$PORT_TO_USE" --proxy-headers --forwarded-allow-ips="*"
