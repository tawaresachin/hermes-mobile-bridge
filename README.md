# Hermes Mobile Bridge

Lightweight REST/SSE backend for the Hermes Mobile Android app.

**Endpoints:**
- `GET /health` — health check
- `POST /api/chat/stream` — SSE streaming chat (OpenAI-compatible)
- `POST /api/chat` — non-streaming chat
- `GET /api/sessions` — list sessions

**Auth:** Bearer token (configurable via `HERMES_API_KEY` env var, default `hermes123`)

**AI Backend:** Any OpenAI-compatible API (default: OpenCode Zen free tier — no key needed)

## Quick Start

```bash
pip install fastapi uvicorn httpx
export HERMES_API_KEY="your-secret-key"
python3 server.py
```

## Config

| Env | Default | Description |
|---|---|---|
| `PORT` | `9119` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `HERMES_API_KEY` | `hermes123` | API key for mobile app auth |
| `AI_BASE_URL` | `https://opencode.ai/zen/v1` | OpenAI-compatible API base |
| `AI_API_KEY` | — | API key for AI provider |
| `AI_MODEL` | `deepseek-v4-flash-free` | Model name |
