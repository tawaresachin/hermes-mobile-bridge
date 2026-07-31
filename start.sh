#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Kill any existing instance
pkill -f "python3.*server.py" 2>/dev/null || true
sleep 1

# Load .env if present (API keys, etc.)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Start bridge server — OmniRoute is the default provider (auto-routing)
export HERMES_API_KEY="${HERMES_API_KEY:?Set HERMES_API_KEY in .env}"
export AI_BASE_URL="${AI_BASE_URL:-http://localhost:20128/v1}"
export AI_MODEL="${AI_MODEL:-auto/best-coding}"
export AI_API_KEY="${AI_API_KEY:-}"

python3 -B server.py >> server.log 2>&1 &
echo $! > server.pid
echo "✅ Hermes Mobile Bridge started (PID: $(cat server.pid))"
echo "   URL: http://0.0.0.0:${PORT:-9119}"
echo "   Provider: ${AI_BASE_URL}"
echo "   Model: ${AI_MODEL}"
