# Hermes Mobile Bridge

**One repo — server + Hermes plugin + auto OS detection (Linux / Windows / macOS / Android).**

Lightweight REST/SSE backend for the Hermes Mobile Android app. Ships with:
- `server.py` — FastAPI bridge (chat stream, sessions, models, TTS, uploads, QR setup)
- `plugin/` — the Hermes plugin (`hermes mobile-serve`)
- `install.py` — **the single command**: installs, updates AND starts everything
- `platform_utils.py` — OS auto-detection: picks the right Python, downloads cloudflared /
  OmniRoute per-OS, detects Tailscale (CLI on desktop, app-mode on Android)
- `tunnel_supervisor.py` — Cloudflare quick-tunnel auto-restart + URL tracking
- `agent_loop.py` — AI + tool orchestration with rolling context summary (JARVIS-style)

## Install, Update & Start — ONE command

```bash
curl -fsSL https://raw.githubusercontent.com/tawaresachin/hermes-mobile-bridge/main/install.py | python3 -
```

Idempotent — run it once on a fresh machine, re-run it any time to update:

| Scenario | What it does |
|---|---|
| **Fresh machine** | installs the Hermes plugin → clones the server → installs deps → starts `hermes mobile-serve` |
| **Re-run (update)** | force-reinstalls the plugin → `git pull` the server → restarts the server |
| **No Hermes CLI** | standalone mode: clones server + deps → runs `python3 server.py` |

Flags: `--no-serve` (install/update only) · `--force` · `--dir DIR` · `--port PORT`.

Requires: Python 3.9+ and git (both needed by Hermes itself anyway).

### Manual (classic two-step)

```bash
hermes plugins install tawaresachin/hermes-mobile-bridge/plugin --enable
hermes mobile-serve
```

`hermes mobile-serve` alone is enough for day-to-day updates afterwards: it
self-syncs the plugin and git-pulls the server on every start
(`BRIDGE_AUTO_UPDATE=0` disables both).

`hermes mobile-serve` on first start writes forced defaults to `.env`:
`CAVEMAN_STYLE=1`, `CONTEXT_RECENT_K=24`, `CONTEXT_SUMMARY_BATCH=12` (caveman
replies + flat token usage on long sessions — every conversation, every restart).

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
| `BRIDGE_AUTO_UPDATE` | `1` | Auto-update plugin + server on `mobile-serve` (set `0` to disable) |
| `CAVEMAN_STYLE` | `1` | Caveman style replies (set `0` to disable) |
| `CONTEXT_RECENT_K` | `24` | Verbatim recent-message window (headroom) |
| `CONTEXT_SUMMARY_BATCH` | `12` | Older messages rolled into the summary |
| `STORE_PATH` | `~/.hermes-mobile-server` | Messages / cache / QR data |

## TTS provider routing

The bridge's `/api/tts` follows **Hermes' TTS config** (`~/.hermes/config.yaml` → `tts:`):

- **`tts.provider` unset or `edge`** (default) → built-in edge-tts with the app's
  9-language voice selector (`en-IN-NeerjaNeural`, `hi-IN-SwaraNeural`, …)
- **`tts.provider` set to anything else** (`elevenlabs`, `openai`, `xai`, `minimax`,
  `gemini`, `mistral`, `deepinfra`, `piper`, `neutts`, `kittentts`, or a custom
  `type: command` provider) → synthesis is delegated to Hermes' own
  `text_to_speech_tool`, so premium voices work by configuring Hermes normally.
  Unknown providers fall back to edge inside Hermes' tool (fail-open).

## Security

- Path traversal neutralized: session IDs are SHA-256 hashed before use in filenames
- TTS voice comes from a fixed enum on the app side (no injection surface)
- Auth: Bearer `HERMES_API_KEY` or JWT (register/login endpoints)
