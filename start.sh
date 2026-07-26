#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Kill any existing instance
pkill -f "python3.*server.py" 2>/dev/null || true
sleep 1

# Start bridge server
export HERMES_API_KEY="${HERMES_API_KEY:-hermes123}"
export AI_BASE_URL="${AI_BASE_URL:-https://opencode.ai/zen/v1}"
export AI_MODEL="${AI_MODEL:-deepseek-v4-flash-free}"

python3 -B server.py >> server.log 2>&1 &
echo $! > server.pid
echo "✅ Hermes Mobile Bridge started (PID: $(cat server.pid))"
echo "   URL: http://0.0.0.0:${PORT:-9119}"
echo "   API Key: ${HERMES_API_KEY}"
