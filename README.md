# Hermes Mobile Bridge

**One repo — server + Hermes plugin + auto OS detection (Linux / Windows / macOS / Android).**

Lightweight REST/SSE backend for the Hermes Mobile Android app. Ships with:
- `server.py` — FastAPI bridge (chat stream, sessions, models, TTS, uploads, QR setup)
- `plugin/` — the Hermes plugin (`hermes mobile-serve`) — installed by symlink, zero copy-drift
- `platform_utils.py` — OS auto-detection: picks the right Python, downloads cloudflared /
  OmniRoute per-OS, detects Tailscale (CLI on desktop, app-mode on Android)
- `tunnel_supervisor.py` — Cloudflare quick-tunnel auto-restart + URL tracking
- `agent_loop.py` — AI + tool orchestration with rolling context summary (JARVIS-style)

## Install (any OS)

```bash
git clone https://github.com/tawaresachin/hermes-mobile-bridge
cd hermes-mobile-bridge
pip install -r requirements.txt          # or: pip install fastapi uvicorn httpx bcrypt pyjwt email-validator sse-starlette
export HERMES_API_KEY="your-secret-key"
python3 server.py                        # standalone
```

### With Hermes (recommended — fully automatic)

```bash
hermes plugins install tawaresachin/hermes-mobile-bridge/plugin --enable
hermes mobile-serve
```

**That's it — two commands on any OS.** What happens automatically:

1. `plugins install .../plugin --enable` clones the repo, installs the `mobile-bridge`
   plugin (subdir-aware installer), and enables it
2. `hermes mobile-serve` **bootstraps the server itself** if missing (auto-clone +
   `pip install -r requirements.txt`), then: OS auto-detect → OmniRoute → Tailscale →
   Cloudflare tunnel fallback → forced defaults → server up

`hermes mobile-serve` on first start writes forced defaults to `.env`:
`CAVEMAN_STYLE=1`, `CONTEXT_RECENT_K=24`, `CONTEXT_SUMMARY_BATCH=12` (caveman
replies + flat token usage on long sessions — every conversation, every restart).

### Updating

**Server + bridge code** — automatic. Every `hermes mobile-serve` start runs
`git pull` on the managed checkout (`BRIDGE_AUTO_UPDATE=0` disables). New
server features go live on the next restart — no manual step.

**Plugin** — one command (Hermes subdir installs keep no `.git`, so reinstall
is the update path):

```bash
hermes plugins install tawaresachin/hermes-mobile-bridge/plugin --enable --force
```

(If you cloned the server repo manually instead of letting it bootstrap,
`git -C ~/hermes-mobile-server pull` also works.)

### Standalone (no Hermes)

```bash
git clone https://github.com/tawaresachin/hermes-mobile-bridge
cd hermes-mobile-bridge
pip install -r requirements.txt
export HERMES_API_KEY="your-secret-key"
python3 server.py
```

## OS auto-detection

| OS | Python | Tailscale | cloudflared | OmniRoute |
|---|---|---|---|---|
| Android/Termux | `/data/data/com.termux/.../python3` | app-mode (netlink needs root) | linux-arm64 binary | linux binary |
| Linux | `python3` | `tailscale ip -4` CLI | linux-amd64 binary | linux binary |
| macOS | `/opt/homebrew/bin/python3` | `tailscale ip -4` CLI | darwin .tgz | darwin binary |
| Windows | `py -3` / `python` | `tailscale ip -4` CLI | .exe | .exe |

No Tailscale? Cloudflare tunnel is the automatic fallback (kept fresh by the supervisor).

## Endpoints

- `GET /health` — health check
- `POST /api/chat/stream` — SSE streaming chat (OpenAI-compatible)
- `POST /api/chat` — non-streaming chat
- `GET /api/sessions` — list sessions
- `GET /api/models` — live model list (never hardcoded)
- `POST /api/models/switch` — switch active model (session or global)
- `POST /api/tts` — edge-tts synthesis (per-language voices, cached)
- `GET /setup/qr` — QR pairing for the app

## Config

| Env | Default | Description |
|---|---|---|
| `PORT` | `9119` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `HERMES_API_KEY` | `hermes123` | API key for mobile app auth |
| `AI_BASE_URL` | `http://localhost:20128/v1` | OpenAI-compatible API base |
| `AI_MODEL` | `auto/best-coding` | Active model |
| `TTS_VOICE` | `en-IN-NeerjaNeural` | Default edge-tts voice |
| `CAVEMAN_STYLE` | `1` | Caveman style replies (set `0` to disable) |
| `CONTEXT_RECENT_K` | `24` | Verbatim recent-message window (headroom) |
| `CONTEXT_SUMMARY_BATCH` | `12` | Older messages rolled into the summary |
| `STORE_PATH` | `~/.hermes-mobile-server` | Messages / cache / QR data |

## Security

- Path traversal neutralized: session IDs are SHA-256 hashed before use in filenames
- TTS voice comes from a fixed enum on the app side (no injection surface)
- Auth: Bearer `HERMES_API_KEY` or JWT (register/login endpoints)
