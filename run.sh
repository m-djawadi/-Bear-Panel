#!/usr/bin/env bash
# Convenience launcher for the Bear Panel platform.
# Starts the FastAPI backend; open frontend/index.html in a browser separately.
set -euo pipefail
cd "$(dirname "$0")/backend"

if [ ! -d ".venv" ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

echo "Backend starting on http://127.0.0.1:8000"
echo "Now open frontend/index.html in your browser."
python local_swarm.py
