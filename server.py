#!/usr/bin/env python3
"""
Hermes Mobile Bridge Server
REST/SSE backend that the Hermes Mobile Android app connects to.
Forwards chat to OpenCode Zen (or any OpenAI-compatible API).

New in v2: User authentication with JWT + refresh tokens.
"""
from __future__ import annotations

import asyncio
import hmac
import hashlib
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import concurrent.futures as _cf
# Shared executor for TTS worker pipe reads (see _read_exact deadlock note).
_TTS_READ_POOL = _cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts-read")
import urllib.request
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Depends, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field

# Local auth modules
from auth_db import get_auth_db
from auth_handler import (
    get_jwt_secret,
    create_access_token,
    decode_access_token,
    generate_refresh_token,
    register_user,
    login_user,
    refresh_tokens,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hermes-bridge")

# ─── Load .env if present ────────────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _val = _line.split("=", 1)
            _val = _val.strip("\"'")
            if _key not in os.environ:
                os.environ[_key] = _val
            logger.info("Loaded .env: %s", _key)
else:
    logger.info("No .env file found at %s", _env_path)

# ─── Config ──────────────────────────────────────────────────────────────

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9119"))

# AI provider (OpenAI-compatible) — Omnirouter default for auto-routing
AI_BASE_URL = os.getenv("AI_BASE_URL", "http://localhost:20128/v1")
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "auto/best-coding")

# Session-scoped model overrides (live switching, no restart needed)
# Keyed by session_id, value is model name
_session_model_overrides: dict[str, str] = {}
# Maps model_id → base_url for provider-aware switching
_model_base_url_map: dict[str, str] = {}
_session_base_url_overrides: dict[str, str] = {}
# Maps model_id → API key for the model's provider (provider-aware auth).
# Populated by list_models() from env keys / config.yaml custom providers.
_model_api_key_map: dict[str, str] = {}
_session_api_key_overrides: dict[str, str] = {}


def _resolve_model(session_id: str) -> str:
    """Get the effective model for a session: override if set, else default."""
    return _session_model_overrides.get(session_id, AI_MODEL)


def _resolve_base_url(session_id: str) -> str:
    """Get the effective base URL for a session."""
    return _session_base_url_overrides.get(session_id, AI_BASE_URL)


def _resolve_api_key(session_id: str) -> str:
    """Get the effective API key for a session.

    Provider-aware: if the session switched to a model from another provider
    (OpenCode Zen, OpenRouter, HF Router, custom), return THAT provider's key.
    Falls back to the default AI_API_KEY (OmniRoute) when no override exists.
    """
    # 1. Explicit per-session override (set on model switch)
    key = _session_api_key_overrides.get(session_id, "")
    if key:
        return key
    # 2. Look up the session's current model in the provider map
    model = _resolve_model(session_id)
    key = _model_api_key_map.get(model, "")
    if key:
        return key
    # 3. Default (OmniRoute / main provider)
    return AI_API_KEY


# Bounded-session helper: caps an override dict at MAX_SESSION_OVERRIDES
# entries (oldest evicted) so memory can't grow without limit.
MAX_SESSION_OVERRIDES = 200


# ─── Persistent session model overrides ─────────────────────────────────
# Session switches live in-memory for routing speed, but a server restart
# used to wipe them silently — the app then showed the DEFAULT model again
# ("I switched but it shows old"). Persist {sid: {model, base_url}} and
# re-resolve API keys at boot from config/env (keys are never stored).

def _session_overrides_path() -> Path:
    return STORE_PATH / "session_overrides.json"


def _save_session_overrides() -> None:
    try:
        data = {
            sid: {
                "model": model,
                "base_url": _session_base_url_overrides.get(sid, ""),
            }
            for sid, model in _session_model_overrides.items()
        }
        _session_overrides_path().write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning("Failed to save session overrides: %s", e)


def _load_session_overrides() -> None:
    try:
        p = _session_overrides_path()
        if not p.exists():
            return
        data = json.loads(p.read_text())
        for sid, info in data.items():
            if not isinstance(info, dict):
                continue
            model = info.get("model", "")
            bu = info.get("base_url", "")
            if model:
                _session_model_overrides[sid] = model
            if bu:
                _session_base_url_overrides[sid] = bu
                key = _runtime_key_for_base_url(bu)
                if key:
                    _session_api_key_overrides[sid] = key
        if data:
            logger.info("Restored %d session model override(s)", len(data))
    except Exception as e:
        logger.warning("Failed to load session overrides: %s", e)


def _bounded_override(d: dict, key: str) -> None:
    if len(d) >= MAX_SESSION_OVERRIDES:
        # Evict the oldest inserted key (dicts preserve insertion order)
        try:
            d.pop(next(iter(d)))
        except (StopIteration, KeyError):
            pass


def _runtime_key_for_base_url(base_url: str) -> str:
    """Auto-resolve the API key for a provider base URL at runtime.

    ZERO hardcoded values — every source is discovered live:
      1. config.yaml custom_providers whose base_url prefix-matches,
         with ${ENV_VAR} interpolation in api_key values.
      2. Environment scan: any *_API_KEY / *_TOKEN / *_KEY var whose name
         contains a host segment of the base URL (e.g. OPENCODE_ZEN_API_KEY
         matches opencode.ai, OPENROUTER_API_KEY matches openrouter.ai).
      3. config.yaml auxiliary.<task>.api_key entries whose base_url
         prefix-matches (e.g. auxiliary.vision → HF Router key).
    Returns "" when nothing matches — callers fall back to AI_API_KEY.
    """
    import re as _re
    bl = (base_url or "").lower().rstrip("/")
    if not bl:
        return ""

    def _interp(v: str) -> str:
        return _re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), v or "")

    # Lazy import — hermes_features reads the config from the Hermes home dir
    from hermes_features import _read_config_yaml
    config = _read_config_yaml()

    # 1. custom_providers prefix match (config-driven, env-interpolated)
    raw_cp = config.get("custom_providers", {})
    cp_list = raw_cp if isinstance(raw_cp, list) else [raw_cp]
    for cp in cp_list:
        if not isinstance(cp, dict):
            continue
        bu = (cp.get("base_url") or "").lower().rstrip("/")
        if bu and bl.startswith(bu):
            key = _interp(cp.get("api_key", ""))
            if key:
                return key

    # 2. auxiliary.<task>.api_key prefix match (e.g. vision → HF Router).
    #    Declared config wins over env heuristics — checked BEFORE the env
    #    scan so a generic host segment (e.g. "router" in openrouter) can't
    #    shadow a provider that's explicitly configured.
    for task, task_cfg in config.get("auxiliary", {}).items():
        if not isinstance(task_cfg, dict):
            continue
        bu = (task_cfg.get("base_url") or "").lower().rstrip("/")
        if bu and bl.startswith(bu):
            key = _interp(task_cfg.get("api_key", ""))
            if key:
                return key

    # 3. Environment scan — match host segments against env var names.
    host = bl.split("://")[-1].split("/")[0].split(":")[0]  # e.g. opencode.ai
    segments = [s for s in host.split(".") if s.isalpha() and len(s) >= 4]
    if segments:
        best_var, best_len = "", 0
        for name, val in sorted(os.environ.items()):
            if not val:
                continue
            if not (name.endswith("_API_KEY") or name.endswith("_TOKEN") or name.endswith("_KEY")):
                continue
            name_l = name.lower()
            # Score = longest host segment found in this env var name
            score = max((len(s) for s in segments if s in name_l), default=0)
            if score > best_len:
                best_var, best_len = name, score
        if best_var:
            return os.environ[best_var]

    return ""


def _handle_special_command(query: str, session_id: str) -> tuple[bool, str | None]:
    """
    Handle commands that need to run BEFORE the agent loop (model switching, etc.)
    Returns (handled, response_text). If handled=True, the caller should return
    the response immediately without creating an AgentLoop.
    """
    q = query.strip()
    if not q:
        return False, None

    parts = q.split()
    cmd = parts[0].lower()

    # ─── Live model switch (/model <name> [--global]) ───
    if cmd == "/model" and len(parts) >= 2 and parts[1] not in ("help", "?", "list"):
        is_global = "--global" in parts or "-g" in parts
        # Pick the first non-flag token as the model name (flags can appear anywhere)
        model_name = next((p for p in parts[1:] if p not in ("--global", "-g")), "")
        if not model_name:
            return False, None

        # Bounded caches: keep only the most recent N sessions so the
        # override dicts can't grow unbounded (memory hygiene).
        _bounded_override(_session_model_overrides, session_id)
        if not is_global:
            _bounded_override(_session_base_url_overrides, session_id)
            _bounded_override(_session_api_key_overrides, session_id)

        if is_global:
            # Write to .env as new default (persistent) — atomic: temp file + rename
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                content = env_path.read_text(encoding="utf-8")
                if "AI_MODEL=" in content:
                    lines = content.splitlines()
                    content = "\n".join(
                        f"AI_MODEL={model_name}" if l.startswith("AI_MODEL=") else l
                        for l in lines
                    )
                else:
                    content += f"\nAI_MODEL={model_name}"
                tmp_path = env_path.with_suffix(".env.tmp")
                tmp_path.write_text(content, encoding="utf-8")
                tmp_path.replace(env_path)  # atomic on POSIX
            # Also update the module-level default for immediate effect
            import server as _srv_mod
            _srv_mod.AI_MODEL = model_name
            # Global switch → also update global base URL + API key defaults
            if model_name in _model_base_url_map:
                _srv_mod.AI_BASE_URL = _model_base_url_map[model_name]
                key = _model_api_key_map.get(model_name) or _runtime_key_for_base_url(_model_base_url_map[model_name])
                if key:
                    _srv_mod.AI_API_KEY = key
            return True, f"✅ Model permanently switched to `{model_name}` (global)."
        else:
            # Session-scoped override (live, no restart)
            _session_model_overrides[session_id] = model_name
            # Also switch base URL + API key if we know them for this model
            if model_name in _model_base_url_map:
                _session_base_url_overrides[session_id] = _model_base_url_map[model_name]
            if model_name in _model_api_key_map:
                _session_api_key_overrides[session_id] = _model_api_key_map[model_name]
            elif model_name in _model_base_url_map:
                # Fall back to runtime key inference from the base URL
                key = _runtime_key_for_base_url(_model_base_url_map[model_name])
                if key:
                    _session_api_key_overrides[session_id] = key
            # Persist so the switch survives server restarts (and the app
            # never sees the model "revert to old" after a reboot).
            _save_session_overrides()
            return True, f"✅ Model switched to `{model_name}` for this session."

    return False, None

# Optional static API key for backward compatibility
# If set, accepts Authorization: Bearer <this_key> as fallback
HERMES_API_KEY = os.getenv("HERMES_API_KEY", "")

# Data storage
STORE_PATH = Path(os.getenv("STORE_PATH", os.path.expanduser("~/.hermes-mobile-server")))
STORE_PATH.mkdir(parents=True, exist_ok=True)

# Uploads directory — created eagerly so StaticFiles can mount it
UPLOADS_DIR = STORE_PATH / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Initialize auth DB
auth_db = get_auth_db(STORE_PATH)
JWT_SECRET = get_jwt_secret(STORE_PATH)
# Sweep expired refresh tokens at boot — otherwise the table grows forever
# (every login/refresh inserts; only the presented token is deleted).
try:
    auth_db.cleanup_expired_tokens()
except Exception as _e:
    logger.warning("refresh-token cleanup failed at boot: %s", _e)

# ─── Data store (sessions & messages) ────────────────────────────────────


def _sessions_path() -> Path:
    return STORE_PATH / "sessions.json"


def _messages_path(session_id: str) -> Path:
    # SECURITY: session_id is client-controlled — never allow path traversal.
    # Only safe chars pass through; anything else is scrubbed.
    safe = re.sub(r"[^A-Za-z0-9_-]", "", session_id or "")
    return STORE_PATH / f"messages_{safe}.json"


def _valid_session_id(session_id: str) -> bool:
    """Reject malformed session ids at the API boundary.

    Sanitizing (as _messages_path does) would let a request for
    ``victim!`` collide with the victim's ``messages_victim.json`` —
    the ownership gate checks the RAW id, so the collision bypasses it.
    Rejecting the id outright closes that hole (same rule /api/upload uses)."""
    return bool(session_id) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id) is not None


# Serializes load-modify-save on sessions.json / messages_*.json. Without
# it, two concurrent turns on the same session both read, both write — one
# update is silently lost and messageCount regresses.
_FS_LOCK = threading.Lock()

# ─── Detached generation task cap ───────────────────────────────────────
# Burst disconnects must not stack unbounded agent runs (each holds an
# LLM stream, tool subprocesses and response buffers). A semaphore gates
# concurrent runs; a done-callback discards the reference + frees the slot.
_AGENT_TASK_SEM = asyncio.Semaphore(4)
_AGENT_TASKS: set["asyncio.Task"] = set()


def load_sessions() -> list[dict]:
    p = _sessions_path()
    if p.exists():
        return json.loads(p.read_text())
    return []


def save_sessions(sessions: list[dict]) -> None:
    _sessions_path().write_text(json.dumps(sessions, indent=2, default=str))


def load_messages(session_id: str) -> list[dict]:
    p = _messages_path(session_id)
    if p.exists():
        return json.loads(p.read_text())
    return []


def save_messages(session_id: str, msgs: list[dict]) -> None:
    # Memory + disk bound: keep only the most recent N messages per session.
    # Older turns already rolled into the agent's rolling summary, and the
    # app's chat view shows recent history — trimming caps per-request
    # load_messages() cost (read + parse + rewrite was O(n) and unbounded).
    MAX_STORED_MESSAGES = 300
    if len(msgs) > MAX_STORED_MESSAGES:
        msgs = msgs[-MAX_STORED_MESSAGES:]
    _messages_path(session_id).write_text(json.dumps(msgs, indent=2, default=str))


# ─── Shared helpers ─────────────────────────────────────────────────────


def _save_user_message(session_id: str, query: str, attachment_url: str = "", attachment_type: str = "") -> None:
    """Append the user's query to the message history and persist."""
    with _FS_LOCK:
        msgs = load_messages(session_id)
        entry = {"role": "user", "content": query, "timestamp": time.time()}
        if attachment_url:
            entry["attachment_url"] = attachment_url
            entry["attachment_type"] = attachment_type
        msgs.append(entry)
        save_messages(session_id, msgs)


def _save_assistant_message(session_id: str, content: str, reasoning_content: str = "") -> None:
    """Append the assistant response to the message history and persist.
    reasoning_content (DeepSeek thinking mode) is stored so it can be
    echoed back to the API on the next turn — DeepSeek requires it."""
    with _FS_LOCK:
        msgs = load_messages(session_id)
        entry = {"role": "assistant", "content": content, "timestamp": time.time()}
        if reasoning_content:
            entry["reasoning_content"] = reasoning_content
        msgs.append(entry)
        save_messages(session_id, msgs)


def _upsert_session(session_id: str, query: str, owner_id: int | None = None) -> None:
    """Create or update a session entry from a user query."""
    with _FS_LOCK:
        sessions = load_sessions()
        existing = next((s for s in sessions if s["id"] == session_id), None)
        if existing:
            existing["messageCount"] = (existing.get("messageCount", 0) or 0) + 1
            existing["updatedAt"] = int(time.time() * 1000)
            # Legacy sessions (pre-ownership) get claimed by the first user who
            # touches them — auto-migrates old data without exposing it to others.
            if owner_id is not None and existing.get("owner_id") is None:
                existing["owner_id"] = owner_id
        else:
            sessions.insert(
                0,
                {
                    "id": session_id,
                    "title": query[:60] + ("…" if len(query) > 60 else ""),
                    "messageCount": 1,
                    "createdAt": int(time.time() * 1000),
                    "updatedAt": int(time.time() * 1000),
                    "owner_id": owner_id,
                },
            )
        save_sessions(sessions)


def _session_owner_id(session_id: str) -> int | None:
    """Owner (user_id) of a session, or None if it doesn't exist yet."""
    for s in load_sessions():
        if s.get("id") == session_id:
            return s.get("owner_id")
    return None


def _can_use_session(session_id: str, user_id: int) -> bool:
    """Ownership gate: a session may be used only by its owner.

    Unknown sessions (not yet in the store) are claimable — the caller
    creates them, and _upsert_session stamps the owner on creation.
    Sessions owned by someone else are off-limits (403)."""
    owner = _session_owner_id(session_id)
    if owner is None:
        return True  # new session — caller will create + claim it
    return owner == user_id


_ATTACH_TEXT_CACHE: dict[str, tuple[float, int, str]] = {}  # path -> (mtime, size, snippet)


def _read_attachment_content(attach_url: str, attach_type: str) -> str:
    """Try to read uploaded file content from the local filesystem.
    Returns a string snippet to embed in the message, or empty string if unreadable.

    Results are cached by (path, mtime, size) — attachments are effectively
    immutable, and OCR costs up to 30s per image per turn otherwise."""
    if not attach_url or not attach_url.startswith("/uploads/"):
        return ""

    try:
        # Parse /uploads/{session_id}/{filename}
        parts = attach_url.split("/")
        if len(parts) < 4:
            return ""
        sess = parts[2]
        fname = "/".join(parts[3:])
        if not _valid_session_id(sess):
            return ""
        fpath = UPLOADS_DIR / sess / fname
        # Path-traversal guard: resolved path MUST stay inside the uploads dir
        resolved = fpath.resolve()
        if not str(resolved).startswith(str(UPLOADS_DIR.resolve())) or ".." in parts[2:]:
            return ""
        if not resolved.exists() or not resolved.is_file():
            return ""

        st = None
        try:
            st = resolved.stat()
            cached = _ATTACH_TEXT_CACHE.get(str(resolved))
            if cached and cached[0] == st.st_mtime and cached[1] == st.st_size:
                return cached[2]
        except OSError:
            st = None

        def _store(snippet: str) -> str:
            if st is not None:
                _ATTACH_TEXT_CACHE[str(resolved)] = (st.st_mtime, st.st_size, snippet)
                if len(_ATTACH_TEXT_CACHE) > 256:  # bounded
                    _ATTACH_TEXT_CACHE.pop(next(iter(_ATTACH_TEXT_CACHE)))
            return snippet

        # Read first 8KB of content
        raw = resolved.read_bytes()[:8192]

        # Try to decode as text
        text = None
        for enc in ("utf-8", "latin-1"):
            try:
                decoded = raw.decode(enc)
                # Verify it's readable text (not binary pretending to be latin-1)
                printable = sum(1 for c in decoded if c.isprintable() or c in "\n\r\t")
                ratio = printable / max(len(decoded), 1)
                if ratio > 0.85:
                    text = decoded
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if text is None:
            # Binary file — try OCR if it's an image
            size = fpath.stat().st_size
            mime_hint = attach_type or "unknown"
            
            if attach_type and attach_type.startswith("image/"):
                # Try OCR with tesseract (Marathi + English)
                try:
                    ocr_text = subprocess.run(
                        ["tesseract", str(resolved), "stdout", "-l", "mar+eng", "--psm", "6"],
                        capture_output=True, text=True, timeout=30
                    ).stdout.strip()
                    if ocr_text and len(ocr_text) > 10:
                        return _store(f"\n\n--- Attached image (OCR extracted): {fname} ---\n{ocr_text}\n--- End of OCR text ---")
                except Exception:
                    pass
            
            return _store(f"[{mime_hint} file, {size} bytes — content not readable as text]")

        # Truncate to reasonable length
        if len(text) > 6000:
            text = text[:6000] + "\n… [truncated]"

        return _store(f"\n\n--- Attached file: {fname} ---\n{text}\n--- End of attached file ---")

    except Exception as e:
        logger.warning("Failed to read attachment %s: %s", attach_url, e)
        return ""


def _inject_reply_context(openai_messages: list[dict], reply_to: str | None) -> None:
    """Attach a Telegram-style reply quote to the last user message.
    Prompt-only context — never persisted to history. The last message is
    the just-saved query (single user message after sanitization)."""
    if not reply_to or not reply_to.strip():
        return
    for m in reversed(openai_messages):
        if m["role"] == "user":
            m["content"] = f'[Replying to your previous message: "{reply_to[:500]}"]\n\n{m["content"]}'
            return


def _build_openai_messages(session_id: str) -> list[dict]:
    """Build the conversation history array for the OpenAI-compatible API.
    Reads uploaded file content from local storage and embeds it as text.
    Echoes back reasoning_content (DeepSeek thinking mode) — DeepSeek
    requires it on follow-up turns or it returns 400.
    Does NOT include a system prompt — AgentLoop adds its own.

    Sanitizes poisoned history: blank replies used to be saved as '' (the
    pre-fix server skipped nothing), leaving user,user,user chains that
    violate role alternation — deepseek answers 200 with EMPTY content.
    Empty entries are dropped and consecutive same-role entries merged.
    """
    msgs = load_messages(session_id)
    result = []
    for m in msgs:
        content = m["content"] or ""
        role = m["role"]
        attach_url = m.get("attachment_url", "")
        attach_type = m.get("attachment_type", "")

        # Drop empty entries (blank assistant replies from the pre-fix era)
        # — an empty message breaks strict role alternation.
        if not content.strip() and not m.get("reasoning_content"):
            continue

        if attach_url:
            file_snippet = _read_attachment_content(attach_url, attach_type)
            enhanced = content
            if file_snippet:
                enhanced += file_snippet
            else:
                label = "Image" if attach_type and "image" in attach_type else "File"
                enhanced += f"\n\n[{label} attached: {attach_url}]"
            entry: dict = {"role": role, "content": enhanced}
            if m.get("reasoning_content"):
                entry["reasoning_content"] = m["reasoning_content"]
        else:
            entry = {"role": role, "content": content}
            # Echo back DeepSeek's reasoning_content from prior assistant turns
            if m.get("reasoning_content"):
                entry["reasoning_content"] = m["reasoning_content"]

        # Collapse consecutive same-role entries (the blank-reply bug left
        # user,user,user chains) — keep the first, merge the text.
        if result and result[-1]["role"] == entry["role"]:
            prev = result[-1]
            prev_content = prev.get("content") or ""
            if content.strip():
                prev["content"] = f"{prev_content}\n\n{content}" if prev_content.strip() else content
            if not prev.get("reasoning_content") and entry.get("reasoning_content"):
                prev["reasoning_content"] = entry["reasoning_content"]
            continue
        result.append(entry)

    # Drop orphan tool messages (a tool result with no matching assistant
    # tool_calls entry — old malformed runs could leave those too).
    cleaned = []
    pending_ids: set = set()
    for e in result:
        if e["role"] == "assistant" and e.get("tool_calls"):
            pending_ids = {tc["id"] for tc in e["tool_calls"]}
            cleaned.append(e)
        elif e["role"] == "tool":
            if e.get("tool_call_id") in pending_ids:
                cleaned.append(e)
            # orphan — drop silently
        else:
            cleaned.append(e)
    return cleaned


# ─── Models ──────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str = Field(max_length=8_000)
    session_id: Optional[str] = Field(default=None, max_length=64)
    stream: bool = True
    attachment_url: Optional[str] = Field(default=None, max_length=512)
    attachment_type: Optional[str] = Field(default=None, max_length=32)
    multi_agent: bool = False
    # Telegram-style reply: quoted text of the message being replied to.
    # Injected into the model prompt (NOT persisted in history).
    reply_to: Optional[str] = Field(default=None, max_length=2_000)


class TtsRequest(BaseModel):
    text: str = Field(max_length=4_000)
    voice: Optional[str] = Field(default=None, max_length=64)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(max_length=72)  # bcrypt truncates at 72 bytes


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


# ─── App ─────────────────────────────────────────────────────────────────

VERSION = "2.2.1"  # single source of truth for the bridge version

app = FastAPI(title="Hermes Mobile Bridge", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    # Native app doesn't send Origin (no browser CORS) — localhost allows dev tooling.
    # Credentials + wildcard is spec-invalid, so use explicit localhost origins.
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static file mount for uploaded files ────────────────────────────────

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

# ─── Auth dependency ────────────────────────────────────────────────────


async def verify_bearer(request: Request) -> dict:
    """
    FastAPI dependency: validate Authorization: Bearer ***
    Returns the decoded JWT payload (contains sub=user_id, email).
    Also accepts the static HERMES_API_KEY as fallback.
    Raises 401 on failure.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth[7:].strip()

    # Try JWT first
    payload = decode_access_token(token, JWT_SECRET)
    if payload:
        # Verify user still exists
        try:
            user_id = int(payload["sub"])
            user = auth_db.get_user_by_email(payload["email"])
            if not user or user.id != user_id:
                raise HTTPException(status_code=401, detail="User not found")
        except (ValueError, KeyError):
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return payload

    # Fallback: check static API key (backward compat) — timing-safe
    if HERMES_API_KEY and hmac.compare_digest(token, HERMES_API_KEY):
        return {"sub": "0", "email": "legacy@api-key", "type": "legacy"}

    raise HTTPException(status_code=401, detail="Invalid or expired token")


# Public endpoints that don't require auth
PUBLIC_PATHS = {"/health", "/diag", "/auth/register", "/auth/login", "/auth/refresh", "/setup", "/setup/qr", "/setup/connect"}


def _uploads_allowed(request: Request, user: dict) -> bool:
    """Attachment fetch gate: /uploads/<session>/... only for the session owner.

    Runs inside auth_middleware so the static /uploads mount stays behind
    per-session ownership — an authenticated user cannot fetch another
    user's attachments by guessing the URL.

    TRAVERSAL GUARD: Starlette's StaticFiles collapses ``..`` segments when
    resolving the file, so checking the RAW path segment alone is a bypass
    (``/uploads/mine/../victim/file``). We normalize the path FIRST and check
    the real session segment, and reject any path containing ``..`` outright.
    """
    import posixpath
    path = request.url.path
    if not path.startswith("/uploads/"):
        return True
    if ".." in path:
        return False
    parts = posixpath.normpath(path.strip("/")).split("/")
    if len(parts) < 2:
        return False
    try:
        uid = int(user.get("sub", "-1"))
    except (ValueError, TypeError):
        uid = -1
    return _can_use_session(parts[1], uid)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Skip auth for public paths; enforce for everything else."""
    # NOTE: /uploads is intentionally NOT whitelisted — uploaded attachments
    # (transcripts, images) require auth; the app sends Bearer via Coil.
    if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/auth/"):
        return await call_next(request)
    # Check auth
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return Response(status_code=401, content='{"detail":"Missing or invalid Authorization header"}', media_type="application/json")
    token = auth[7:].strip()
    
    # Try JWT first (new auth)
    payload = decode_access_token(token, JWT_SECRET)
    if payload:
        request.state.user = payload
        if not _uploads_allowed(request, payload):
            return Response(status_code=401, content='{"detail":"Not your attachment"}', media_type="application/json")
        return await call_next(request)
    
    # Fallback: check if it's the static HERMES_API_KEY (backward compat)
    if HERMES_API_KEY and hmac.compare_digest(token, HERMES_API_KEY):
        request.state.user = {"sub": "0", "email": "legacy@api-key", "type": "legacy"}
        if not _uploads_allowed(request, request.state.user):
            return Response(status_code=401, content='{"detail":"Not your attachment"}', media_type="application/json")
        return await call_next(request)
    
    # Neither worked
    return Response(status_code=401, content='{"detail":"Invalid or expired token"}', media_type="application/json")


def _is_authorized(request: Request) -> bool:
    """Check a request carries valid auth: JWT, the bridge API key, or a session token."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return False
    token = auth[7:].strip()
    if decode_access_token(token, JWT_SECRET):
        return True
    return bool(HERMES_API_KEY and hmac.compare_digest(token, HERMES_API_KEY))


# ─── Auth rate limiting (brute-force / spam protection) ───
_auth_attempts: dict[str, list[float]] = {}
_auth_attempts_lock = threading.Lock()
AUTH_RATE_LIMIT = 20          # max attempts per window
AUTH_RATE_WINDOW = 900        # 15 minutes


def _client_ip(request: Request) -> str:
    """Client IP — honour X-Forwarded-For ONLY when the direct peer is
    loopback (i.e. the request arrived via the local cloudflared tunnel,
    which sets the header itself). External clients connecting straight
    to the server can spoof X-Forwarded-For — never trust it then, or the
    auth rate limit is trivially bypassed by rotating the header."""
    peer = request.client.host if request.client else "unknown"
    if peer in ("127.0.0.1", "::1"):
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    return peer


def _auth_rate_limited(request: Request) -> bool:
    """True if the client has exceeded the auth attempt budget."""
    ip = _client_ip(request)
    now = time.time()
    with _auth_attempts_lock:
        # Opportunistic purge: bounds the dict (an IP that never returns
        # would otherwise leave its entry forever).
        if len(_auth_attempts) > 10_000:
            for k, v in list(_auth_attempts.items()):
                if not [t for t in v if now - t < AUTH_RATE_WINDOW]:
                    _auth_attempts.pop(k, None)
        hits = [t for t in _auth_attempts.get(ip, []) if now - t < AUTH_RATE_WINDOW]
        if len(hits) >= AUTH_RATE_LIMIT:
            _auth_attempts[ip] = hits
            return True
        hits.append(now)
        _auth_attempts[ip] = hits
        return False


# ─── One-time account claim tokens (QR pairing → sign in as the user) ───
# After registering/logging in on the web setup page, the server issues a
# short-lived token. The page shows a QR containing it; the app scans and
# exchanges it for a JWT for THAT user — never a password in QR.
# Tokens are PERSISTED to disk so a server restart (or the 15-min TTL) can't
# silently invalidate a QR the user is about to scan.
_claim_tokens: dict[str, dict] = {}
_claim_lock = threading.Lock()
CLAIM_TTL = 900  # 15 minutes (user preference — not longer)
_CLAIMS_FILE = STORE_PATH / "claims.json"


def _claims_save() -> None:
    try:
        _CLAIMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CLAIMS_FILE.write_text(json.dumps(_claim_tokens))
    except Exception:
        pass  # best-effort; in-memory still works for the session


def _claims_load() -> None:
    global _claim_tokens
    try:
        if _CLAIMS_FILE.exists():
            _claim_tokens = json.loads(_CLAIMS_FILE.read_text()) or {}
            # Drop anything already expired.
            now = time.time()
            _claim_tokens = {t: e for t, e in _claim_tokens.items() if e.get("expires", 0) > now}
    except Exception:
        _claim_tokens = {}


def _issue_claim(user_id: int) -> str:
    token = secrets.token_urlsafe(24)
    with _claim_lock:
        # Opportunistic purge of expired entries (bounds memory + disk).
        now = time.time()
        expired = [t for t, e in _claim_tokens.items() if e["expires"] < now]
        for t in expired:
            _claim_tokens.pop(t, None)
        _claim_tokens[token] = {"user_id": user_id, "expires": now + CLAIM_TTL}
        _claims_save()
    return token


def _validate_claim(token: str) -> Optional[int]:
    """Validate a claim token WITHOUT consuming it.

    Multi-use until expiry (15 min): the user may re-scan the same QR after
    logging out — it always signs back into the SAME registered account.
    The token is bound to a user id and time-boxed, so replay within the
    window only ever re-authenticates that user (same exposure as the
    password, minus the time limit)."""
    if not token:
        return None
    with _claim_lock:
        entry = _claim_tokens.get(token)
    if not entry or entry["expires"] < time.time():
        return None
    return entry["user_id"]


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    """Structured access log with duration — one line per request.
    Registered after auth_middleware, so it wraps it (401s are logged too)."""
    start = time.perf_counter()
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("req=%s unhandled error on %s %s", request_id, request.method, request.url.path)
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "req=%s %s %s -> %d (%.1fms)",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    return response


@app.post("/auth/register")
async def auth_register(request: Request, body: RegisterRequest):
    """Register a new user. Returns access + refresh tokens.
    Rate-limited per client IP to prevent spam accounts."""
    if _auth_rate_limited(request):
        raise HTTPException(status_code=429, detail="Too many attempts — try again later")
    try:
        result = await register_user(STORE_PATH, body.email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # One-time claim token so the web setup page can produce a "sign in as
    # this user" QR for the app (no password ever goes into a QR).
    result["claim_token"] = _issue_claim(int(result.get("user_id", 0)))
    return result


@app.post("/auth/login")
async def auth_login(request: Request, body: LoginRequest):
    """Log in with email + password. Returns access + refresh tokens.
    Rate-limited per client IP to slow brute-force attempts."""
    if _auth_rate_limited(request):
        raise HTTPException(status_code=429, detail="Too many attempts — try again later")
    result = await login_user(STORE_PATH, body.email, body.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    result["claim_token"] = _issue_claim(int(result.get("user_id", 0)))
    return result


class ClaimRequest(BaseModel):
    token: str


@app.post("/auth/claim")
async def auth_claim(body: ClaimRequest):
    """Exchange a claim token (from the pairing QR) for a JWT.
    Reusable until its 15-minute expiry, so re-scanning after logout signs
    back into the SAME registered account. Lets the app sign in as the user
    who registered on the web setup page — without ever carrying a password."""
    user_id = _validate_claim(body.token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid or expired claim token")
    # Locked methods — the shared sqlite connection must only be touched
    # under @_synchronized.
    email = auth_db.get_user_email(user_id)
    if not email:
        raise HTTPException(status_code=401, detail="User not found")
    secret = get_jwt_secret(STORE_PATH)
    access = create_access_token(user_id, email, secret)
    refresh_raw, _ = auth_db.create_refresh_token(user_id)
    return {"token": access, "refresh_token": refresh_raw, "user_id": str(user_id), "email": email}


@app.post("/auth/refresh")
async def auth_refresh(body: RefreshRequest):
    """Rotate refresh token. Returns new access + refresh tokens."""
    result = await refresh_tokens(STORE_PATH, body.refresh_token)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    return result


# ─── Core Endpoints ─────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/diag")
async def diag(request: Request):
    """Diagnostic page — open this in a browser on your phone to test connectivity."""
    real_ip = _client_ip(request)
    return {
        "status": "ok",
        "message": "Hermes Mobile Bridge is reachable from your phone! ✅",
        "your_ip": real_ip,
        "server_time": time.ctime(),
        "endpoints": {
            "health": "/health",
            "chat": "POST /api/chat",
            "chat_stream": "POST /api/chat/stream",
            "sessions": "GET /api/sessions",
            "auth_register": "POST /auth/register",
            "auth_login": "POST /auth/login",
            "auth_refresh": "POST /auth/refresh",
            "diag_log": "POST /api/diag/log",
        },
    }


class DiagLogRequest(BaseModel):
    device: str = Field(default="", max_length=64)
    version: str = Field(default="", max_length=32)
    log: str = Field(default="", max_length=300_000)


DIAG_LOGS_DIR = STORE_PATH / "logs"
DIAG_LOGS_MAX_FILES = 20


@app.post("/api/diag/log")
async def upload_diag_log(body: DiagLogRequest, user: dict = Depends(verify_bearer)):
    """Accept the app's on-device diag.log (last 24h) and store it under
    STORE_PATH/logs/ so the user/maintainer can pull it for analysis.
    Filename is sanitized (device + timestamp); old uploads are pruned
    (keep the newest DIAG_LOGS_MAX_FILES)."""
    try:
        DIAG_LOGS_DIR.mkdir(parents=True, exist_ok=True)
        safe_device = re.sub(r"[^A-Za-z0-9_-]", "_", body.device)[:40] or "device"
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = DIAG_LOGS_DIR / f"{safe_device}-{ts}.txt"
        # Only the user id (sub) goes into the stored file — never the full
        # JWT payload (email etc.).
        user_id = str(user.get("sub", user.get("email", "?")))[:40]
        header = (
            f"# Hermes mobile diag log upload\n"
            f"# device: {body.device}\n"
            f"# version: {body.version}\n"
            f"# uploaded: {time.ctime()}\n"
            f"# user: {user_id}\n"
            f"# ---- last 24h activity ----\n"
        )
        path.write_text(header + body.log, encoding="utf-8", errors="replace")
        # Prune: keep the newest N uploads
        files = sorted(DIAG_LOGS_DIR.glob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
        for stale in files[DIAG_LOGS_MAX_FILES:]:
            try:
                stale.unlink()
            except Exception:
                pass
        logger.info("diag log saved: %s (%d bytes, %d files kept)", path.name, path.stat().st_size, len(files[:DIAG_LOGS_MAX_FILES]))
        return {"status": "ok", "file": path.name}
    except Exception as e:
        logger.warning("diag log upload failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to store diag log")


@app.get("/api/diag/logs")
async def list_diag_logs(uid: int = Depends(verify_bearer)):
    """List stored diag-log uploads (maintainer convenience)."""
    try:
        if not DIAG_LOGS_DIR.exists():
            return {"logs": []}
        files = sorted(DIAG_LOGS_DIR.glob("*.txt"), key=lambda f: f.stat().st_mtime, reverse=True)
        return {
            "logs": [
                {"file": f.name, "size": f.stat().st_size, "modified": f.stat().st_mtime}
                for f in files
            ]
        }
    except Exception as e:
        logger.warning("diag log list failed: %s", e)
        return {"logs": []}


@app.get("/api/sessions")
async def list_sessions(user: dict = Depends(verify_bearer)):
    """List chat sessions for the authenticated user (owner-scoped)."""
    uid = int(user["sub"])
    return [s for s in load_sessions() if s.get("owner_id") == uid]


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user: dict = Depends(verify_bearer)):
    """Fetch a session's stored messages (owner-scoped). Used by the app to
    re-sync the last response when a stream was interrupted client-side
    (tab switch / process death) — the server is the source of truth."""
    uid = int(user["sub"])
    if not _valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if not _can_use_session(session_id, uid):
        return JSONResponse(status_code=403, content={"detail": "Not your session"})
    msgs_path = _messages_path(session_id)
    if not msgs_path.exists():
        return []
    try:
        data = json.loads(msgs_path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(verify_bearer)):
    """Delete a session and its messages (owner-scoped)."""
    uid = int(user["sub"])
    if not _valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    if not _can_use_session(session_id, uid):
        return JSONResponse(status_code=403, content={"detail": "Not your session"})
    sessions = [s for s in load_sessions() if s.get("id") != session_id]
    save_sessions(sessions)
    msgs_path = _messages_path(session_id)
    if msgs_path.exists():
        msgs_path.unlink()
    logger.info("Deleted session %s for user %s", session_id, user["sub"])
    return {"status": "ok", "deleted": session_id}


# ─── TTS (Text-to-Speech) ────────────────────────────────────────────────

# Persistent edge-tts worker: spawned once, reused for every request.
# Protocol: one JSON request per line on stdin; stdout replies with a
# 4-byte big-endian length prefix + raw MP3 (or 0xFFFFFFFF = error marker).
#
# Provider routing: when ~/.hermes/config.yaml sets `tts.provider` to anything
# other than edge, synthesis is delegated to Hermes' own text_to_speech_tool —
# so ElevenLabs / OpenAI / xAI / MiniMax / Gemini / local (piper, neutts) /
# custom command providers all work by just configuring Hermes. Edge remains
# the default (free, 9 Indian languages via the app's voice selector).

_EDGE_TTS_WORKER_SCRIPT = r'''
import asyncio, json, os, sys

# Make Hermes' tool modules importable (worker runs from the hermes venv:
# venv/bin/python3 -> hermes-agent root).
_HERMES_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
if _HERMES_ROOT not in sys.path:
    sys.path.insert(0, _HERMES_ROOT)

import edge_tts

def _configured_provider():
    # The app's voice screen ALWAYS sends edge-tts voice names (Neerja,
    # Aarohi, …), so the bridge defaults to edge-tts REGARDLESS of the
    # shared ~/.hermes/config.yaml (whose provider — e.g. Gemini — has its
    # own quotas and can 429 after 3 requests, breaking voice replies).
    # Opt into Hermes-provider delegation explicitly:
    #   BRIDGE_TTS_PROVIDER=hermes
    env = os.getenv("BRIDGE_TTS_PROVIDER", "").lower()
    return env if env in ("edge", "hermes") else "edge"

def _hermes_synth(text):
    """Delegate to Hermes' text_to_speech_tool (premium / custom providers)."""
    import tempfile
    from tools.tts_tool import text_to_speech_tool
    out = tempfile.mktemp(suffix='.mp3')
    try:
        result = text_to_speech_tool(text, output_path=out)
        data = json.loads(result) if isinstance(result, str) else result
        if data.get('success') and os.path.exists(out):
            audio = open(out, 'rb').read()
            if audio:
                return audio
            raise RuntimeError('hermes TTS produced empty audio')
        raise RuntimeError(str(data.get('error') or 'hermes TTS failed')[:200])
    finally:
        try:
            os.unlink(out)
        except Exception:
            pass

async def _edge_synth(text, voice):
    tts = edge_tts.Communicate(text, voice=voice)
    out = b''
    async for chunk in tts.stream():
        if chunk['type'] == 'audio':
            out += chunk['data']
    return out

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line)
        text = req.get('text', '')
        voice = req.get('voice', 'en-IN-NeerjaNeural')
        # One internal retry — transient network blips.
        for attempt in range(2):
            try:
                if _configured_provider() in ('edge', '', None):
                    audio = asyncio.run(_edge_synth(text, voice))
                else:
                    audio = _hermes_synth(text)
                break
            except Exception:
                if attempt == 0:
                    continue
                raise
        sys.stdout.buffer.write(len(audio).to_bytes(4, 'big') + audio)
    except Exception:
        sys.stdout.buffer.write((0xFFFFFFFF).to_bytes(4, 'big'))
    sys.stdout.buffer.flush()
'''


class _EdgeTtsWorker:
    """Reusable edge-tts subprocess with crash/timeout recovery and a
    one-shot fallback after repeated failures (fail-open)."""

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._r = None      # raw unbuffered reader fd
        self._w = None      # stdin pipe
        self._lock = threading.Lock()
        self._failures = 0
        self._venv_py = str(Path.home() / ".hermes" / "hermes-agent" / "venv" / "bin" / "python3")
        if not Path(self._venv_py).exists():
            self._venv_py = "python3"

    def _spawn(self) -> None:
        self._kill()
        self._proc = subprocess.Popen(
            [self._venv_py, "-c", _EDGE_TTS_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._w = self._proc.stdin
        # Raw, unbuffered reader — select() + os.read() need no buffering
        # layer, otherwise buffer/pipe desync would corrupt the protocol.
        # os.dup() gives an INDEPENDENT fd so closing this wrapper can never
        # invalidate Popen's stdout (avoid "Bad file descriptor" on GC).
        _out = self._proc.stdout
        assert _out is not None, "worker stdout pipe missing"
        self._r = os.fdopen(os.dup(_out.fileno()), "rb", buffering=0)

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2)
            except Exception:
                pass
        # Close the pipe wrappers explicitly BEFORE dropping references —
        # otherwise Python's GC raises "Bad file descriptor" noise on exit.
        if self._r is not None:
            try:
                self._r.close()
            except Exception:
                pass
        if self._w is not None:
            try:
                self._w.close()
            except Exception:
                pass
        self._proc = None
        self._r = None
        self._w = None

    def _read_exact(self, n: int, timeout: float) -> bytes:
        r = self._r
        assert r is not None, "worker not spawned"
        # Thread-based read with timeout — works on Windows too
        # (select() on pipes is POSIX-only).
        # Module-level shared pool: the OLD code created a pool per call and
        # its `with`-exit ran shutdown(wait=True), which blocks FOREVER on a
        # still-pending blocking os.read after a timeout — while holding the
        # global _lock. One hung edge-tts worker wedged ALL TTS permanently.
        buf = b""
        deadline = time.monotonic() + timeout
        while len(buf) < n:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                chunk = _TTS_READ_POOL.submit(
                    lambda: os.read(r.fileno(), n - len(buf))
                ).result(timeout=remaining)
            except Exception:
                raise TimeoutError("TTS worker read timeout") from None
            if not chunk:
                raise EOFError("TTS worker closed")
            buf += chunk
        return buf

    def synth(self, text: str, voice: str, timeout: float = 60.0) -> bytes:
        with self._lock:
            if self._failures >= 3:
                # Repeated worker failures — fail open to the one-shot path
                # (and reset so a later recovery can come back to the worker).
                self._failures = 0
                return self._synth_oneshot(text, voice, timeout)
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._spawn()
                assert self._r is not None and self._w is not None, "worker spawn failed"
                self._w.write(json.dumps({"text": text, "voice": voice}).encode() + b"\n")
                self._w.flush()
                header = self._read_exact(4, timeout)
                length = int.from_bytes(header, "big")
                if length == 0xFFFFFFFF:
                    raise RuntimeError("TTS worker reported error")
                audio = self._read_exact(length, timeout)
                self._failures = 0
                return audio
            except Exception:
                self._failures += 1
                self._kill()
                raise

    def _synth_oneshot(self, text: str, voice: str, timeout: float) -> bytes:
        # Same provider routing as the persistent worker (edge default,
        # Hermes text_to_speech_tool for premium/custom providers).
        script = (
            "import asyncio, json, os, sys\n"
            "_HERMES_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))\n"
            "if _HERMES_ROOT not in sys.path: sys.path.insert(0, _HERMES_ROOT)\n"
            "import edge_tts\n"
            "def _provider():\n"
            "    e = os.getenv('BRIDGE_TTS_PROVIDER', '').lower()\n"
            "    return e if e in ('edge', 'hermes') else 'edge'\n"
            "def _hermes():\n"
            "    import tempfile\n"
            "    from tools.tts_tool import text_to_speech_tool\n"
            "    out = tempfile.mktemp(suffix='.mp3')\n"
            "    try:\n"
            "        r = text_to_speech_tool(sys.argv[1], output_path=out)\n"
            "        d = json.loads(r) if isinstance(r, str) else r\n"
            "        if d.get('success') and os.path.exists(out):\n"
            "            return open(out, 'rb').read()\n"
            "        raise RuntimeError(str(d.get('error') or 'tts failed')[:200])\n"
            "    finally:\n"
            "        try: os.unlink(out)\n"
            "        except Exception: pass\n"
            "async def m():\n"
            "    if _provider() in ('edge', '', None):\n"
            "        tts = edge_tts.Communicate(sys.argv[1], voice=sys.argv[2])\n"
            "        out = b''\n"
            "        async for chunk in tts.stream():\n"
            "            if chunk['type'] == 'audio': out += chunk['data']\n"
            "        return out\n"
            "    return _hermes()\n"
            "sys.stdout.buffer.write(asyncio.run(m()))\n"
        )
        proc = subprocess.run(
            [self._venv_py, "-c", script, text, voice],
            capture_output=True, timeout=timeout,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError(proc.stderr.decode(errors="replace")[:300] or "TTS failed")
        return proc.stdout


_tts_worker = _EdgeTtsWorker()


def _strip_markdown_for_speech(text: str) -> str:
    """Clean markdown/formatting before synthesis so TTS reads naturally
    (JARVIS-style `_preprocess_for_speech` equivalent)."""
    import re
    # Markdown links → link text only
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Bold/italic/code/inline markers
    text = re.sub(r"[*_`~>]", "", text)
    # Heading hashes at line starts
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    # List dashes/bullets/numbering at line starts
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# TTS model cache: {sha256(text+voice) -> mp3 path} so repeat bubbles are
# instant and we don't hammer the TTS provider.
_tts_cache: dict[str, str] = {}


@app.post("/api/tts")
async def text_to_speech(body: TtsRequest, user: dict = Depends(verify_bearer)):
    """Synthesize speech from text (Edge TTS via the Hermes venv).

    Returns MP3 audio bytes. Cached by (text, voice) hash so replaying a
    voice bubble costs nothing. Runs the TTS subprocess on a worker thread
    so the event loop never blocks.
    """
    text = _strip_markdown_for_speech((body.text or "").strip())
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    if len(text) > 4000:
        raise HTTPException(status_code=400, detail="text too long (max 4000 chars)")
    # Default Indian English voice (edge-tts). Override with TTS_VOICE env
    # (e.g. en-IN-PrabhatNeural for male, or any edge-tts voice name).
    voice = (body.voice or "").strip() or os.getenv("TTS_VOICE", "en-IN-NeerjaNeural")

    # Cache hit?
    cache_key = hashlib.sha256(f"{text}::{voice}".encode()).hexdigest()
    cached = _tts_cache.get(cache_key)
    if cached and Path(cached).exists():
        return Response(content=Path(cached).read_bytes(), media_type="audio/mpeg")

    # Synthesize via a PERSISTENT edge-tts worker subprocess (spawned once,
    # reused across requests) so per-sentence TTS doesn't pay Python startup
    # (~300-500ms) on every sentence of a streamed reply. Falls back to a
    # one-shot subprocess after repeated worker failures (fail-open).
    def _synth() -> bytes:
        return _tts_worker.synth(text, voice, timeout=30)

    try:
        audio = await asyncio.to_thread(_synth)
    except Exception as e:
        # One retry — edge-tts network blips (the 502s seen in production
        # logs) are usually transient.
        logger.warning("TTS attempt 1 failed (%s), retrying once", e)
        try:
            audio = await asyncio.to_thread(_synth)
        except Exception as e2:
            logger.error("TTS failed after retry: %s", e2)
            raise HTTPException(status_code=502, detail=f"TTS failed: {e2}")

    # Persist to cache dir
    try:
        cache_dir = STORE_PATH / "tts_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fpath = cache_dir / f"{cache_key}.mp3"
        fpath.write_bytes(audio)
        _tts_cache[cache_key] = str(fpath)
        # Bound the cache dict (memory hygiene) — delete the evicted FILE
        # too, otherwise STORE_PATH grows unbounded (mp3s were never
        # removed, only the dict entry).
        if len(_tts_cache) > 100:
            evicted_key = next(iter(_tts_cache))
            evicted_path = _tts_cache.pop(evicted_key)
            try:
                Path(evicted_path).unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass  # cache is best-effort

    return Response(content=audio, media_type="audio/mpeg")


# ─── Speech-to-text (whisper.cpp) ───────────────────────────────────────
# Runs as a subprocess so the server process never loads the model into
# its own memory (keeps the bridge flat on RAM). One run at a time — the
# pad CPU shouldn't thrash on concurrent voice messages.
STT_BIN = os.getenv("STT_BIN", str(Path.home() / ".hermes-mobile-bridge" / "bin" / "whisper-cli"))
STT_MODEL = os.getenv("STT_MODEL", str(STORE_PATH / "models" / "ggml-base.bin"))
_stt_lock = asyncio.Lock()


@app.post("/api/stt")
async def speech_to_text(request: Request, lang: str = "", user: dict = Depends(verify_bearer)):
    """Transcribe uploaded audio (16 kHz mono WAV) via whisper.cpp.

    Body = raw WAV bytes. Returns {"text": "..."}. Optional ?lang= hint
    (e.g. mr, hi, en); omitted → whisper auto-detects.
    """
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="empty audio body")
    if len(audio) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="audio too large (max 25MB)")
    if not Path(STT_BIN).exists():
        raise HTTPException(status_code=501, detail="whisper-cli not installed (set STT_BIN)")
    if not Path(STT_MODEL).exists():
        raise HTTPException(status_code=501, detail=f"whisper model missing: {STT_MODEL}")

    lang_code = (lang or "").strip().lower()[:2]

    async def _run() -> str:
        async with _stt_lock:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(STORE_PATH)) as tf:
                tf.write(audio)
                tmp_wav = tf.name
            try:
                cmd = [STT_BIN, "-m", STT_MODEL, "-f", tmp_wav, "-otxt", "-t", "4", "-np"]
                if lang_code:
                    cmd += ["-l", lang_code]
                r = await asyncio.to_thread(
                    subprocess.run, cmd, capture_output=True, text=True, timeout=60
                )
                out_txt = Path(tmp_wav + ".txt")
                text = out_txt.read_text(encoding="utf-8").strip() if out_txt.exists() else ""
                out_txt.unlink(missing_ok=True)
                if not text and r.returncode != 0:
                    raise RuntimeError((r.stderr or r.stdout or "whisper failed").strip()[:200])
                return text
            finally:
                try:
                    Path(tmp_wav).unlink(missing_ok=True)
                except Exception:
                    pass

    try:
        text = await asyncio.wait_for(_run(), timeout=75)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="STT timed out")
    except Exception as e:
        logger.warning("STT failed: %s", e)
        raise HTTPException(status_code=502, detail=f"STT failed: {e}")
    return {"text": text}


# ─── Models cache (module-level, disk-persisted + lazy refresh) ───
# Persisted to disk so a server restart NEVER wipes it: the app's model
# picker gets an instant answer at startup. When the cache is stale we
# serve the old list immediately and refresh in the background.
_models_cache: dict = {}
_models_cache_ts: float = 0.0
_models_lock = threading.Lock()
_MODEL_REFRESH_INFLIGHT = False  # single-flight guard for background refresh
_MODELS_CACHE_FILE = STORE_PATH / "models_cache.json"


def _models_save_to_disk() -> None:
    try:
        _MODELS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # Flatten to {ts, models: <list>, current, default, provider} — the
        # same shape the loader normalizes; never nest the response dict.
        models_list = _normalize_models(_models_cache.get("models", []))
        _MODELS_CACHE_FILE.write_text(json.dumps({
            "ts": _models_cache_ts,
            "models": models_list,
            "current": _models_cache.get("current", AI_MODEL),
            "default": _models_cache.get("default", AI_MODEL),
            "provider": _models_cache.get("provider", ""),
        }))
    except Exception as e:
        logger.warning("models cache save failed: %s", e)


def _rebuild_provider_maps(models: list[dict]) -> None:
    """(Re)build model→baseUrl and model→apiKey maps from a models list.
    Keys resolve at runtime from config/env — nothing hardcoded."""
    for m in models:
        if m.get("baseUrl"):
            _model_base_url_map[m["id"]] = m["baseUrl"]
            key = _runtime_key_for_base_url(m["baseUrl"])
            if key:
                _model_api_key_map[m["id"]] = key


def _normalize_models(raw) -> list[dict]:
    """Extract a flat list of model dicts from ANY cache shape:
    list, dict-of-dicts, or the response dict {models: [...], ...}."""
    if isinstance(raw, list):
        return [m for m in raw if isinstance(m, dict)]
    if isinstance(raw, dict):
        if isinstance(raw.get("models"), list):
            return [m for m in raw["models"] if isinstance(m, dict)]
        return [m for m in raw.values() if isinstance(m, dict)]
    return []


def _models_load_from_disk() -> None:
    global _models_cache, _models_cache_ts
    try:
        if _MODELS_CACHE_FILE.exists():
            data = json.loads(_MODELS_CACHE_FILE.read_text())
            if isinstance(data, dict):
                models_raw = data.get("models", [])
                _models_cache_ts = float(data.get("ts", 0.0))
                # current/default/provider may sit at the TOP level (old
                # flattened files) or INSIDE the nested response dict.
                nested = models_raw if isinstance(models_raw, dict) else {}
                current = data.get("current") or nested.get("current") or AI_MODEL
                default = data.get("default") or nested.get("default") or AI_MODEL
                provider = data.get("provider") or nested.get("provider") or ""
            elif isinstance(data, list):
                models_raw, _models_cache_ts = data, 0.0
                current = default = AI_MODEL
                provider = ""
            else:
                models_raw, _models_cache_ts = [], 0.0
                current = default = AI_MODEL
                provider = ""
            models_list = _normalize_models(models_raw)
            _models_cache = {
                "models": models_list,
                "current": current,
                "default": default,
                "provider": provider,
            }
            # CRITICAL: also rebuild the provider maps NOW — otherwise a chat
            # request in the seconds between boot and the background refresh
            # falls back to the DEFAULT API key/base URL and gets 401/400
            # from the session's provider (seen live twice: Agnes 401 after
            # restart, then the local gateway's 'Unable to determine
            # provider' 400).
            _rebuild_provider_maps(models_list)
            logger.info("Loaded %d models from disk cache", len(models_list))
    except Exception as e:
        logger.warning("models cache load failed: %s", e)


def _refresh_models_background(session_id: str) -> None:
    """Thread worker: refetch models, update cache + disk. Never blocks a request.
    Single-flight: if a refresh is already running, skip (N concurrent stale
    /api/models calls used to spawn N provider fetches)."""
    global _models_cache_ts, _MODEL_REFRESH_INFLIGHT
    with _models_lock:
        if _MODEL_REFRESH_INFLIGHT:
            return
        _MODEL_REFRESH_INFLIGHT = True
    try:
        result = _fetch_models_sync(session_id)
        if result:
            with _models_lock:
                _models_cache.clear()
                _models_cache.update(result)
                _models_cache_ts = time.time()
            _models_save_to_disk()
            logger.info("Background model refresh: %d models", len(result))
    except Exception as e:
        logger.warning("background model refresh failed: %s", e)
    finally:
        with _models_lock:
            _MODEL_REFRESH_INFLIGHT = False


@app.get("/api/models")
async def list_models(session_id: str = "", user: dict = Depends(verify_bearer)):
    """List available models — only from providers we can actually query.

    Cache: fresh for 30 min; if stale we SERVE the cached list immediately
    and refresh in the background (the picker never blocks); only a very
    first boot with no cache performs a synchronous fetch.
    """
    global _models_cache_ts
    now = time.time()
    cache_ttl = 1800

    with _models_lock:
        if _models_cache and now - _models_cache_ts < cache_ttl:
            # Cache hit — but "current" must be LIVE per session (a model
            # switch seconds ago must show immediately, not after the
            # 30-min TTL).
            fresh = dict(_models_cache)
            fresh["current"] = _resolve_model(session_id)
            return fresh

    # Stale but present: serve now, refresh in the background.
    if _models_cache:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _refresh_models_background, session_id)
        fresh = dict(_models_cache)
        fresh["current"] = _resolve_model(session_id)
        return fresh

    # No cache at all (first boot): fetch once, synchronously.
    result = await asyncio.to_thread(_fetch_models_sync, session_id)
    if result:
        with _models_lock:
            _models_cache.clear()
            _models_cache.update(result)
            _models_cache_ts = time.time()
        _models_save_to_disk()
    return result


def _fetch_models_sync(session_id: str) -> dict:
    """Blocking model fetch — runs in a worker thread, NOT the event loop.
    External provider queries run in PARALLEL (threads) so a slow provider
    can't stall the others."""
    from hermes_features import _list_available_models, AI_MODEL, AI_BASE_URL, _read_config_yaml
    import urllib.request, urllib.error
    import concurrent.futures

    def _fetch_opencode() -> list[dict]:
        """Query OpenCode/Zen models (no auth needed)."""
        out = []
        try:
            req = urllib.request.Request(
                "https://opencode.ai/zen/v1/models",
                headers={"User-Agent": "HermesMobileBridge/2.0"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                oc_data = json.loads(resp.read())
            for m in oc_data.get("data", []):
                mid = m.get("id", "")
                if mid:
                    out.append({
                        "id": mid,
                        "name": mid,
                        "isFree": mid.endswith("-free"),
                        "isVision": any(k in mid.lower() for k in ("vl", "vision", "multimodal")),
                        "provider": "opencode",
                        "baseUrl": "https://opencode.ai/zen/v1",
                    })
        except Exception as e:
            logger.warning("OpenCode/Zen query failed: %s", e)
        return out

    def _fetch_openrouter() -> list[dict]:
        """Query OpenRouter models if a key is configured."""
        out = []
        or_key = os.environ.get("OPENROUTER_API_KEY")
        if not or_key:
            return out
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {or_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                or_data = json.loads(resp.read())
            for m in or_data.get("data", []):
                mid = m.get("id", "")
                if mid:
                    out.append({
                        "id": mid,
                        "name": mid.split("/")[-1] if "/" in mid else mid,
                        "isFree": mid.endswith(":free") or mid.endswith("-free"),
                        "isVision": any(k in mid.lower() for k in ("vl", "vision", "multimodal")),
                        "provider": "openrouter",
                        "baseUrl": "https://openrouter.ai/api/v1",
                    })
        except Exception as e:
            logger.warning("OpenRouter query failed: %s", e)
        return out

    # Fire the two external queries in parallel
    ext_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(_fetch_opencode)
        f2 = pool.submit(_fetch_openrouter)
        for fut in (f1, f2):
            try:
                ext_results.extend(fut.result())
            except Exception as e:
                logger.warning("Parallel model fetch failed: %s", e)

    seen = set()
    models = []

    # 1. Query current provider — if it's Omnirouter, tag as omnirouter
    raw = _list_available_models()
    is_omnirouter = "localhost:20128" in AI_BASE_URL or "127.0.0.1:20128" in AI_BASE_URL
    provider_label = "omnirouter" if is_omnirouter else None
    for m in raw:
        mid = m.get("id", m) if isinstance(m, dict) else m
        if mid and mid not in seen:
            seen.add(mid)
            prov = provider_label or (mid.split("/")[0] if "/" in mid else "unknown")
            models.append({
                "id": mid,
                "name": mid.split("/")[-1] if "/" in mid else mid,
                "isFree": False,
                "isVision": any(k in mid.lower() for k in ("vl", "vision", "multimodal")),
                "provider": prov,
                "baseUrl": AI_BASE_URL,
            })

    # 2. External providers (already fetched in parallel)
    for m in ext_results:
        mid = m.get("id", "")
        if mid and mid not in seen:
            seen.add(mid)
            models.append(m)

    # 5. Custom providers from config.yaml
    config = _read_config_yaml()
    raw_cp = config.get("custom_providers", {})
    cp_list = raw_cp if isinstance(raw_cp, list) else [raw_cp]
    for cp in cp_list:
        if not isinstance(cp, dict):
            continue
        cp_model = cp.get("model", "")
        cp_name = cp.get("name", "custom")
        cp_base = cp.get("base_url", "")
        cp_models = [cp_model] + list(cp.get("models", []) or [])
        for cp_id in cp_models:
            if not cp_id or cp_id in seen:
                continue
            seen.add(cp_id)
            models.append({
                "id": cp_id,
                "name": cp_id.split("/")[-1] if "/" in cp_id else cp_id,
                "isFree": False,
                "isVision": False,
                "provider": cp_name.lower().replace(" ", "-"),
                "baseUrl": cp_base,
            })

    # ── Normalize flags by model id (consistency across providers) ──
    # The same model can appear from multiple providers (OmniRoute, OpenCode,
    # OpenRouter, custom). Previously each provider set isFree/isVision its own
    # way -> the same model showed a "free" badge in one section but not the
    # other. Single deterministic rule by id: identical id -> identical flags.
    import re as _re
    _free_re = _re.compile(r"(^|[/_:.\-])free($|[/_:.\-])", _re.I)
    _vision_re = _re.compile(r"(vl|vision|multimodal|omni)", _re.I)
    for m in models:
        mid = m.get("id", "")
        m["isFree"] = bool(_free_re.search(mid))
        m["isVision"] = bool(_vision_re.search(mid)) and not m["isFree"]

    # Build base URL + API key maps for provider-aware switching.
    # Keys resolve at runtime from config/env — nothing hardcoded.
    _rebuild_provider_maps(models)

    current = _session_model_overrides.get(session_id, AI_MODEL)
    return {
        "models": models,
        "current": current,
        "default": AI_MODEL,
        "provider": AI_BASE_URL,
    }


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, user: dict = Depends(verify_bearer)):
    """Streaming chat endpoint with full agent tool execution."""
    session_id = body.session_id or uuid.uuid4().hex[:8]
    uid = int(user["sub"])
    if body.session_id and not _valid_session_id(body.session_id):
        return JSONResponse(status_code=400, content={"detail": "Invalid session_id"})
    if not _can_use_session(session_id, uid):
        return JSONResponse(status_code=403, content={"detail": "Not your session"})

    # Reject empty queries (but allow if attachment is present)
    has_attachment = bool(body.attachment_url and body.attachment_url.strip())
    if (not body.query or not body.query.strip()) and not has_attachment:
        return JSONResponse(
            status_code=400,
            content={"detail": "Query cannot be empty"},
        )

    query = body.query.strip() if body.query else ""
    if not query and has_attachment:
        query = "[User attached a file without text]"

    # ─── Pre-check for special commands (model switch, etc.) — BEFORE any
    # persistence: commands are control messages, not chat history, so they
    # must never be saved as user bubbles or bump messageCount.
    handled, cmd_response = _handle_special_command(query, session_id)
    if handled and cmd_response:
        async def _cmd_generator():
            yield f"data: {json.dumps({'type': 'text', 'content': cmd_response})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(
            _cmd_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    _save_user_message(session_id, query, body.attachment_url or "", body.attachment_type or "")
    _upsert_session(session_id, query, uid)

    # Load conversation history — off the event loop: attachment reads / OCR
    # (subprocess, up to 30s) must never block other requests.
    openai_messages = await asyncio.to_thread(_build_openai_messages, session_id)
    _inject_reply_context(openai_messages, body.reply_to)

    # Use session-scoped model override if available
    effective_model = _resolve_model(session_id)

    # Use the agent loop for full tool execution
    from agent_loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
    from hermes_features import set_config
    effective_base_url = _resolve_base_url(session_id)
    set_config(effective_model, effective_base_url)

    loop = AgentLoop(
        ai_base_url=effective_base_url,
        ai_api_key=_resolve_api_key(session_id),
        ai_model=effective_model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        # ── Detached-generation design ──────────────────────────────────
        # The agent runs in a BACKGROUND TASK, NOT in the SSE generator.
        # If the client disconnects (user leaves the chat, app killed, tab
        # switch, network blip), the generator is cancelled — but the task
        # keeps running to completion and SAVES the response. Without this,
        # a disconnect cancelled the agent mid-run and the response was
        # lost server-side (the app's resume-repair could never recover it).
        event_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=64)
        done = asyncio.Event()
        outcome: dict = {"error": None, "text": "", "reasoning": ""}

        async def run_agent() -> None:
            response_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            try:
                async for event in loop.run(
                    openai_messages, query, session_id=session_id,
                    attachment_url=body.attachment_url or "",
                    attachment_type=body.attachment_type or "",
                    multi_agent=body.multi_agent,
                    model_override=_resolve_model(session_id)):
                    # Non-blocking: if the client is gone, drop events rather
                    # than block the task — we still collect + save the text.
                    try:
                        event_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass
                    if event.startswith('data: {'):
                        try:
                            raw = event[6:].strip()
                            data = json.loads(raw)
                            if data.get('type') == 'text':
                                response_chunks.append(data.get('content', ''))
                            elif data.get('type') == 'reasoning':
                                reasoning_chunks.append(data.get('content', ''))
                        except (json.JSONDecodeError, IndexError):
                            pass
                # SAVE INSIDE THE TASK — the generator may be cancelled by a
                # client disconnect at any moment; only the task is guaranteed
                # to reach the save.
                full_text = "".join(response_chunks)
                reasoning = "".join(reasoning_chunks)
                if full_text.strip():
                    logger.info(
                        f"Saving assistant response for session {session_id} "
                        f"({len(full_text)} chars)"
                    )
                    _save_assistant_message(
                        session_id, full_text, reasoning_content=reasoning)
                else:
                    logger.warning(
                        f"Empty full_assistant_response for session {session_id}")
            except asyncio.CancelledError:
                # Task cancelled only at server shutdown — save what we have.
                full_text = "".join(response_chunks)
                if full_text.strip():
                    _save_assistant_message(
                        session_id, full_text,
                        reasoning_content="".join(reasoning_chunks))
            except Exception as e:
                logger.error(f"Stream error for session {session_id}: {e}")
                outcome["error"] = str(e)
            finally:
                done.set()

        await _AGENT_TASK_SEM.acquire()
        try:
            task = asyncio.create_task(run_agent())
        except BaseException:
            _AGENT_TASK_SEM.release()
            raise
        _AGENT_TASKS.add(task)

        def _task_done(t: asyncio.Task) -> None:
            _AGENT_TASKS.discard(t)
            _AGENT_TASK_SEM.release()

        task.add_done_callback(_task_done)
        try:
            # Stream to the client while the task runs. On disconnect the
            # generator is cancelled — the task survives and saves.
            while not done.is_set():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)
                    yield event
                except asyncio.TimeoutError:
                    continue
            # Drain any events produced in the final instant
            while not event_queue.empty():
                yield event_queue.get_nowait()

            if outcome["error"] is not None:
                friendly = "Agent error while generating response. Try again or switch model."
                yield f"data: {json.dumps({'type': 'error', 'content': f'⚠ {friendly}'})}\n\n"
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            # Client disconnected — do NOT cancel the agent task; it keeps
            # running and saves the response. Re-raise to close the stream.
            raise
        finally:
            # If we got here cleanly, reap the task; if cancelled, the task
            # outlives this generator by design (detached generation).
            if not task.done():
                # Give it a moment; do NOT cancel on client disconnect.
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=0.1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat_sync(body: ChatRequest, user: dict = Depends(verify_bearer)):
    """Non-streaming chat endpoint with full agent tool execution."""
    session_id = body.session_id or uuid.uuid4().hex[:8]
    uid = int(user["sub"])
    if body.session_id and not _valid_session_id(body.session_id):
        return JSONResponse(status_code=400, content={"detail": "Invalid session_id"})
    if not _can_use_session(session_id, uid):
        return JSONResponse(status_code=403, content={"detail": "Not your session"})

    has_attachment = bool(body.attachment_url and body.attachment_url.strip())
    if (not body.query or not body.query.strip()) and not has_attachment:
        return JSONResponse(status_code=400, content={"detail": "Query cannot be empty"})

    query = body.query.strip() if body.query else ""
    if not query and has_attachment:
        query = "[User attached a file without text]"

    # ─── Special commands first — never persisted (control, not chat) ───
    handled, cmd_response = _handle_special_command(query, session_id)
    if handled and cmd_response:
        return {"response": cmd_response, "session_id": session_id}

    _save_user_message(session_id, query, body.attachment_url or "", body.attachment_type or "")
    _upsert_session(session_id, query, uid)

    # Use session-scoped model override if available
    effective_model = _resolve_model(session_id)
    effective_base_url = _resolve_base_url(session_id)

    # Use agent loop but just collect the final response
    from agent_loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
    from hermes_features import set_config as _set_config
    _set_config(effective_model, effective_base_url)

    loop = AgentLoop(
        ai_base_url=effective_base_url,
        ai_api_key=_resolve_api_key(session_id),
        ai_model=effective_model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    openai_messages = await asyncio.to_thread(_build_openai_messages, session_id)
    _inject_reply_context(openai_messages, body.reply_to)
    # List + join: O(n) — string += in a loop is O(n²) for long responses
    response_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    final_response = ""
    error_msg = None

    try:
        async for event in loop.run(openai_messages, query, session_id=session_id,
                                    attachment_url=body.attachment_url or "",
                                    attachment_type=body.attachment_type or "",
                                    multi_agent=body.multi_agent,
                                    model_override=_resolve_model(session_id)):
            # Try to parse every SSE data event (cover both text and error types)
            if event.startswith('data: {'):
                try:
                    raw = event[6:].strip()
                    data = json.loads(raw)
                    if data.get('type') == 'text':
                        response_chunks.append(data.get('content', ''))
                    elif data.get('type') == 'reasoning':
                        reasoning_chunks.append(data.get('content', ''))
                    elif data.get('type') == 'error':
                        error_msg = data.get('content', 'Agent error')
                except (json.JSONDecodeError, IndexError):
                    pass
    except Exception as e:
        # Friendly error — never leak raw exception text to the client
        # (the stream path sanitizes; the sync path must too).
        logger.warning("sync chat failed for session %s: %s", session_id, e)
        error_msg = "Request failed. Check connection and try again."

    final_response = "".join(response_chunks)

    if error_msg:
        return {"response": f"⚠️ {error_msg}", "session_id": session_id}

    if final_response.strip():
        _save_assistant_message(
            session_id, final_response,
            reasoning_content="".join(reasoning_chunks),
        )
    else:
        final_response = f"⚠️ No response generated"

    return {"response": final_response, "session_id": session_id}


# ─── File Upload Endpoint ──────────────────────────────────────────────


@app.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(verify_bearer),
):
    """Upload a file to the session's uploads directory."""
    # JWT auth via the shared dependency (header only — query-param tokens
    # leak into access logs). Static-key users pass through here too,
    # matching every other endpoint.
    payload = user

    # Validate user still exists (skip for legacy static-key payloads —
    # they have no DB row by design; every other endpoint lets them pass).
    user_id = int(payload["sub"])
    if payload.get("type") != "legacy":
        user_row = auth_db.get_user_by_email(payload["email"])
        if not user_row or user_row.id != user_id:
            raise HTTPException(status_code=401, detail="User not found")

    # Determine session_id from query param, or use user-specific default
    session_id = request.query_params.get("session_id", payload.get("sub", "default"))

    # Path-traversal guard: session_id must be a plain identifier (no separators)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id or ""):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # Ownership gate: can't upload into another user's session.
    if not _can_use_session(session_id, user_id):
        raise HTTPException(status_code=403, detail="Not your session")

    # Ensure the session subdirectory exists
    session_upload_dir = UPLOADS_DIR / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename — strip directory separators
    raw_filename = file.filename or "untitled"
    safe_filename = os.path.basename(raw_filename)
    if not safe_filename:
        safe_filename = "untitled"

    # Resolve full path, avoid overwrite by appending a suffix if needed
    dest_path = (session_upload_dir / safe_filename).resolve()
    # Belt-and-braces: resolved path must stay inside the session upload dir
    if not str(dest_path).startswith(str(UPLOADS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if dest_path.exists():
        stem, ext = os.path.splitext(safe_filename)
        counter = 1
        while dest_path.exists():
            dest_path = (session_upload_dir / f"{stem}_{counter}{ext}").resolve()
            counter += 1

    # Save the file — streamed in chunks with a size cap so a huge upload
    # can never balloon memory (whole-file read would double it in RAM).
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
    written = 0
    try:
        with dest_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB at a time
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_SIZE:
                    out.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")
                out.write(chunk)
    except HTTPException:
        raise
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Upload failed")
    finally:
        await file.close()

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(safe_filename)
    mime_type = mime_type or file.content_type or "application/octet-stream"

    logger.info(
        "Uploaded file %s (%d bytes, %s) for session %s",
        safe_filename,
        written,
        mime_type,
        session_id,
    )

    return {
        "url": f"/uploads/{session_id}/{dest_path.name}",
        "filename": dest_path.name,
        "mime_type": mime_type,
    }


@app.get("/download/apk")
async def download_apk():
    """Download the latest Hermes Mobile APK."""
    apk_path = STORE_PATH / "uploads" / "hermes-mobile.apk"
    if not apk_path.exists():
        raise HTTPException(status_code=404, detail="APK not found. Build it first with ./gradlew assembleDebug")
    return FileResponse(
        path=str(apk_path),
        media_type="application/vnd.android.package-archive",
        filename="hermes-mobile.apk",
        headers={
            "Content-Disposition": 'attachment; filename="hermes-mobile.apk"',
            "Cache-Control": "no-cache",
        },
    )


# ─── QR Code Setup Endpoints ─────────────────────────────────────────

# One-time setup token — required to fetch /setup/qr and /setup/connect.
# Carried inside the QR payload so scanning works, but random internet
# scanners hitting the public tunnel get 401 instead of the API key.
SETUP_TOKEN = os.getenv("SETUP_TOKEN") or secrets.token_urlsafe(12)


SETUP_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hermes Mobile Bridge — Setup</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: radial-gradient(1200px 600px at 50% -10%, #1b1038 0%, #0a0618 55%, #050310 100%);
    min-height: 100vh; color: #e8e6f5; display: flex; align-items: center; justify-content: center; padding: 24px;
  }
  .card {
    background: rgba(20, 14, 44, 0.75); border: 1px solid rgba(140, 90, 255, 0.25);
    border-radius: 20px; padding: 36px 32px; max-width: 430px; width: 100%;
    box-shadow: 0 20px 60px rgba(0,0,0,0.55); backdrop-filter: blur(8px);
  }
  h1 { font-size: 22px; font-weight: 700; letter-spacing: 0.2px; }
  .sub { color: #a99fd6; font-size: 14px; margin-top: 6px; margin-bottom: 24px; }
  .step { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
  .step-num {
    width: 26px; height: 26px; border-radius: 50%; background: linear-gradient(135deg, #8a3bff, #5b21d6);
    color: #fff; font-weight: 700; font-size: 14px; display: flex; align-items: center; justify-content: center;
  }
  .step-label { font-size: 15px; font-weight: 600; }
  .field { margin-bottom: 12px; }
  .field label { display: block; font-size: 12.5px; color: #a99fd6; margin-bottom: 6px; }
  .field input {
    width: 100%; padding: 11px 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.14);
    background: rgba(0,0,0,0.35); color: #e8e6f5; font-size: 14px; outline: none;
  }
  .pw-row { position: relative; }
  .pw-row input { padding-right: 42px; }
  .eye {
    position: absolute; right: 6px; top: 50%; transform: translateY(-50%);
    background: none; border: 0; cursor: pointer; color: #8d84b5; padding: 6px;
  }
  .eye svg { width: 18px; height: 18px; display: block; }
  .field input:focus { border-color: #8a3bff; }
  .btn {
    display: block; width: 100%; text-align: center; background: linear-gradient(135deg, #8a3bff, #5b21d6);
    color: #fff; font-weight: 600; padding: 13px; border-radius: 12px; font-size: 15px; border: 0; cursor: pointer; margin-top: 4px;
  }
  .btn:hover { filter: brightness(1.12); }
  .link { text-align: center; color: #8a3bff; font-size: 13.5px; margin-top: 14px; cursor: pointer; background: none; border: 0; width: 100%; }
  .link:hover { text-decoration: underline; }
  .msg { font-size: 13px; margin-top: 12px; text-align: center; min-height: 18px; }
  .msg.ok { color: #5ce0a8; }
  .msg.err { color: #ff8f8f; }
  .qr-wrap {
    background: #fff; border-radius: 16px; padding: 14px; width: 252px; height: 252px;
    margin: 0 auto 24px; display: flex; align-items: center; justify-content: center;
  }
  .qr-wrap img { width: 224px; height: 224px; image-rendering: pixelated; }
  .ok-email { text-align: center; color: #5ce0a8; font-weight: 600; margin-bottom: 18px; word-break: break-all; }
  .info {
    background: rgba(0,0,0,0.35); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 12px 14px; font-family: ui-monospace, Menlo, Consolas, monospace;
    font-size: 12.5px; word-break: break-all; color: #b9b0e0; margin-bottom: 20px;
  }
  ol { margin: 0 0 8px 18px; font-size: 14px; line-height: 1.9; color: #cfc9ec; }
  .hint { text-align: center; color: #7d74a8; font-size: 12.5px; margin-top: 14px; }
</style>
</head>
<body>
  <div class="card">
    <h1>🐝 Hermes Mobile Bridge</h1>
    <div class="sub">Set up your phone app — 2 steps</div>

    <!-- STEP 1: account (shown first) -->
    <div id="step1">
      <div class="step"><div class="step-num">1</div><div class="step-label" id="step1title">Create your account</div></div>
      <form id="regform" autocomplete="off">
        <div class="field">
          <label for="email">Email</label>
          <input type="email" id="email" name="email" required placeholder="you@example.com" autocomplete="email">
        </div>
        <div class="field">
          <label for="pw">Password (min 8 characters)</label>
          <div class="pw-row">
            <input type="password" id="pw" name="pw" required minlength="8" placeholder="••••••••" autocomplete="new-password">
            <button type="button" class="eye" data-target="pw" aria-label="Show password"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></button>
          </div>
        </div>
        <div class="field" id="pw2field">
          <label for="pw2">Confirm password</label>
          <div class="pw-row">
            <input type="password" id="pw2" name="pw2" required minlength="8" placeholder="••••••••" autocomplete="new-password">
            <button type="button" class="eye" data-target="pw2" aria-label="Show password"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg></button>
          </div>
        </div>
        <button type="submit" class="btn" id="actbtn">Create account</button>
        <button type="button" class="link" id="togbtn">Already registered? Log in</button>
      </form>
      <div class="msg" id="msg1"></div>
      <div class="info" style="margin-top:16px"><b>Server</b> {base_url}</div>
    </div>

    <!-- STEP 2: claim QR (shown only AFTER account exists) -->
    <div id="step2" style="display:none">
      <div class="step"><div class="step-num">2</div><div class="step-label">Scan to connect your phone</div></div>
      <div class="ok-email" id="okemail"></div>
      <div class="qr-wrap"><img id="qr" alt="Sign-in QR code"></div>
      <ol>
        <li>Open the <b>Hermes</b> app on your phone</li>
        <li>Go to <b>Settings</b> and tap <b>Scan QR Code</b></li>
        <li>Point the camera at this QR code</li>
        <li>Done — the app connects and signs in as your account</li>
      </ol>
      <div class="hint" style="margin-top:10px">The QR expires in 15 minutes. If it runs out, tap “Register / log in another account” below and sign in again.</div>
      <button type="button" class="btn" id="backbtn">← Register / log in another account</button>
      <div class="msg" id="msg2"></div>
    </div>
    <div class="hint">The QR is generated only after your account exists — it contains a one-time sign-in token, never a password.</div>
  </div>
<script>
  var PAGE_TOKEN = "{token}";
  var regMode = true;
  function showStep(which) {
    document.getElementById("step1").style.display = (which === 1) ? "block" : "none";
    document.getElementById("step2").style.display = (which === 2) ? "block" : "none";
  }
  document.getElementById("togbtn").addEventListener("click", function () {
    regMode = !regMode;
    document.getElementById("step1title").textContent = regMode ? "Create your account" : "Log in";
    document.getElementById("pw2field").style.display = regMode ? "block" : "none";
    // FIX: a hidden field with `required` still blocks form submission.
    // Disabling it in login mode lets the submit event actually fire.
    document.getElementById("pw2").disabled = !regMode;
    document.getElementById("actbtn").textContent = regMode ? "Create account" : "Log in";
    document.getElementById("msg1").className = "msg"; document.getElementById("msg1").textContent = "";
  });
  document.querySelectorAll(".eye").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var inp = document.getElementById(btn.dataset.target);
      inp.type = (inp.type === "password") ? "text" : "password";
    });
  });
  document.getElementById("backbtn").addEventListener("click", function () { showStep(1); });
  document.getElementById("regform").addEventListener("submit", async function (e) {
    e.preventDefault();
    var email = document.getElementById("email").value.trim();
    var pw = document.getElementById("pw").value;
    var pw2 = document.getElementById("pw2").value;
    var msg = document.getElementById("msg1");
    var btn = document.getElementById("actbtn");
    if (regMode && pw !== pw2) { msg.className = "msg err"; msg.textContent = "Passwords do not match"; return; }
    msg.className = "msg"; msg.textContent = "Please wait…";
    btn.disabled = true;
    try {
      var path = regMode ? "/auth/register" : "/auth/login";
      var r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: email, password: pw })
      });
      var data = await r.json();
      if (r.ok && data.claim_token) {
        document.getElementById("okemail").textContent = "Signed in as " + data.email;
        document.getElementById("qr").setAttribute("src", "/setup/qr?token=" + encodeURIComponent(PAGE_TOKEN) + "&claim=" + encodeURIComponent(data.claim_token));
        showStep(2);
      } else {
        msg.className = "msg err";
        msg.textContent = (data.detail || "Request failed") + "";
      }
    } catch (err) {
      msg.className = "msg err";
      msg.textContent = "Network error — is the server reachable?";
    }
    btn.disabled = false;
  });
</script>
</body>
</html>"""




@app.get("/setup")
async def setup_page(request: Request, token: str = ""):
    """Styled HTML page that DISPLAYS the pairing QR code (in-browser UI)."""
    if not (hmac.compare_digest(token, SETUP_TOKEN) or _is_authorized(request)):
        raise HTTPException(status_code=401, detail="Invalid setup token")
    base_url = str(request.base_url).rstrip("/")
    page = SETUP_PAGE_HTML.replace("{token}", token or SETUP_TOKEN).replace("{base_url}", base_url)
    return HTMLResponse(
        page,
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@app.get("/setup/qr")
async def setup_qr(request: Request, token: str = "", claim: str = ""):
    """Return a QR code PNG that the mobile app scans to auto-configure.

    With ?claim=<one-time token> the QR additionally carries the claim so
    scanning signs the app into the user who registered on the web page."""
    # Accept: ?token=SETUP_TOKEN (QR pairing) OR valid JWT / bridge key (app session)
    if not (hmac.compare_digest(token, SETUP_TOKEN) or _is_authorized(request)):
        raise HTTPException(status_code=401, detail="Invalid setup token")
    # Validate an optionally provided claim token (must be un-expired).
    claim_ok = False
    if claim:
        with _claim_lock:
            entry = _claim_tokens.get(claim)
        claim_ok = bool(entry and entry["expires"] >= time.time())
        if not claim_ok:
            raise HTTPException(status_code=401, detail="Invalid or expired claim token")
    import io
    try:
        import qrcode
        from qrcode.image.pil import PilImage
    except ImportError:
        raise HTTPException(status_code=501, detail="qrcode[pil] not installed — run: pip install qrcode[pil]")

    # Tailscale IP (primary) → tunnel URL (fallback) → LAN IP (last resort)
    ts_ip = await _detect_host_ip_async()
    if ts_ip and ts_ip.startswith("100."):
        setup_url = f"hermes://connect?host={ts_ip}&port={PORT}&key={HERMES_API_KEY}&setup={SETUP_TOKEN}"
    else:
        tunnel_url = _detect_tunnel_url()
        if tunnel_url:
            setup_url = f"hermes://connect?url={tunnel_url}&key={HERMES_API_KEY}&setup={SETUP_TOKEN}"
        else:
            host_ip = ts_ip or "127.0.0.1"
            setup_url = f"hermes://connect?host={host_ip}&port={PORT}&key={HERMES_API_KEY}&setup={SETUP_TOKEN}"
    if claim_ok:
        setup_url += f"&claim={claim}"

    img = qrcode.make(setup_url, image_factory=PilImage)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@app.get("/setup/connect")
async def setup_connect(request: Request, token: str = ""):
    """Return connection JSON for the app to auto-configure.
    Route-aware: replies with the URL matching the route the caller used
    (tunnel caller → current live tunnel URL; Tailscale caller → 100.x),
    so a refresh never switches an app to an unreachable address."""
    # Accept: ?token=SETUP_TOKEN (QR pairing) OR valid JWT / bridge key (app session)
    if not (hmac.compare_digest(token, SETUP_TOKEN) or _is_authorized(request)):
        raise HTTPException(status_code=401, detail="Invalid setup token")

    # Which route did the caller use to reach us?
    host_header = (request.headers.get("host") or "").lower()

    # 1. Caller came through the Cloudflare tunnel → give them the CURRENT
    #    live tunnel URL (kept fresh by tunnel_supervisor.py).
    if "trycloudflare.com" in host_header:
        tunnel_url = _detect_tunnel_url()
        if tunnel_url:
            return {
                "url": tunnel_url,
                "key": HERMES_API_KEY,
                "model": AI_MODEL,
                "provider": AI_BASE_URL,
                "version": VERSION,
            }

    # 2. Caller used the Tailscale IP directly → keep them on Tailscale.
    ts_ip = await _detect_host_ip_async()
    if ts_ip and ts_ip.startswith("100."):
        return {
            "host": ts_ip,
            "port": PORT,
            "url": f"http://{ts_ip}:{PORT}",
            "key": HERMES_API_KEY,
            "model": AI_MODEL,
            "provider": AI_BASE_URL,
            "version": VERSION,
        }

    # 3. Fallback: tunnel → LAN IP (last resort)
    tunnel_url = _detect_tunnel_url()
    if tunnel_url:
        return {
            "url": tunnel_url,
            "key": HERMES_API_KEY,
            "model": AI_MODEL,
            "provider": AI_BASE_URL,
            "version": VERSION,
        }
    host_ip = await _detect_host_ip_async()
    return {
        "host": host_ip,
        "port": PORT,
        "url": f"http://{host_ip}:{PORT}",
        "key": HERMES_API_KEY,
        "model": AI_MODEL,
        "provider": AI_BASE_URL,
        "version": VERSION,
    }


_host_ip_cache: tuple[float, str] | None = None
_HOST_IP_CACHE_TTL = 600  # seconds — the IP rarely changes


async def _detect_host_ip_async() -> str:
    """Cached, event-loop-safe wrapper for _detect_host_ip.

    The sync version blocks on subprocess probes (up to ~15s); running it
    on the event loop froze setup endpoints. to_thread + 10-min cache."""
    global _host_ip_cache
    now = time.time()
    if _host_ip_cache and now - _host_ip_cache[0] < _HOST_IP_CACHE_TTL:
        return _host_ip_cache[1]
    ip = await asyncio.to_thread(_detect_host_ip)
    _host_ip_cache = (now, ip)
    return ip


def _detect_tunnel_url() -> str:
    """Detect the active Cloudflare tunnel URL, if any."""
    tunnel_url = os.getenv("HERMES_TUNNEL_URL", "")
    if not tunnel_url:
        url_file = STORE_PATH / ".current_tunnel_url"
        if url_file.exists():
            content = url_file.read_text()
            match = re.search(r'https://[a-z0-9.-]+\.trycloudflare\.com', content)
            if match:
                tunnel_url = match.group(0)
    return tunnel_url


def _detect_host_ip() -> str:
    """Detect the best IP for the app to connect to.

    WARNING: runs up to 3 sequential subprocess probes (~5s each) — call
    via _detect_host_ip_async() from async handlers, never directly."""
    # 0. Env override (set by plugin's Tailscale auto-setup)
    ts_ip = os.environ.get("HERMES_TAILSCALE_IP", "")
    if ts_ip and ts_ip.startswith("100."):
        return ts_ip

    # 0b. Config-file override (~/.hermes-mobile-server/tailscale-ip)
    # Android Termux can't run `ip addr` (netlink needs root), so we support
    # a manually-written file: `echo 100.x.y.z > tailscale-ip`
    try:
        ts_file = STORE_PATH / "tailscale-ip"
        if ts_file.exists():
            ip = ts_file.read_text().strip()
            if ip and ip.startswith("100."):
                return ip
    except Exception:
        pass

    # 1. Tailscale IP (primary — via tailscale0 interface or CLI)
    try:
        result = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=5,
        )
        ip = result.stdout.strip()
        if ip and ip.startswith("100."):
            return ip
    except Exception:
        pass

    # 1b. Detect Tailscale IP from network interfaces (Android app mode)
    try:
        result = subprocess.run(
            ["ip", "addr"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "100." in line and "inet " in line:
                match = re.search(r"inet (100\.\d+\.\d+\.\d+)", line)
                if match:
                    return match.group(1)
    except Exception:
        pass

    # 2. LAN IP (WiFi)
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip:
            return ip
    except Exception:
        pass

    # 3. Fallback
    return "127.0.0.1"

# Discovery is optional — only runs when HERMES_REGISTRY_URL is set


async def start_discovery():
    """Initialize identity and start heartbeat loop."""
    registry_url = os.getenv("HERMES_REGISTRY_URL")
    if not registry_url:
        print("   No HERMES_REGISTRY_URL set — skipping registry registration")
        return

    from discovery import BridgeIdentity, RegistryClient

    identity = BridgeIdentity().load_or_create()

    if not identity.is_loaded():
        # First run — prompt for email
        email = os.getenv("HERMES_EMAIL", "")
        if not email:
            print("⚠ HERMES_EMAIL not set. Set it to register with the discovery registry.")
            return
        identity.create_new(email)
        print(f"   Registered new identity: {identity.device_id[:16]}...")

    client = RegistryClient(identity)
    await client.register()

    # Detect tunnel URL
    tunnel_url = os.getenv("HERMES_TUNNEL_URL", "")
    if not tunnel_url:
        # Try to detect from cloudflared log file
        url_file = STORE_PATH / ".current_tunnel_url"
        if url_file.exists():
            content = url_file.read_text()
            match = re.search(r'https://[a-z0-9.-]+\.trycloudflare\.com', content)
            if match:
                tunnel_url = match.group(0)

    if tunnel_url:
        await client.heartbeat(
            tunnel_url=tunnel_url,
            platform=sys.platform,
            version=VERSION,
        )
        print(f"   Tunnel URL: {tunnel_url}")
    else:
        print("⚠ No tunnel URL detected — set HERMES_TUNNEL_URL or start cloudflared")

    # Start periodic heartbeat
    async def heartbeat_loop():
        while True:
            await asyncio.sleep(180)
            try:
                # Re-read the tunnel URL on EVERY beat — the supervisor
                # rewrites .current_tunnel_url when cloudflared rotates the
                # URL; the startup snapshot would go stale.
                current = tunnel_url or ""
                try:
                    st = STORE_PATH / ".current_tunnel_url"
                    if st.exists():
                        current = st.read_text().strip()
                except Exception:
                    pass
                url = os.getenv("HERMES_TUNNEL_URL", current or "")
                if url:
                    await client.heartbeat(url, platform=sys.platform)
            except Exception:
                pass

    asyncio.create_task(heartbeat_loop())


@app.on_event("startup")
async def on_startup():
    await start_discovery()
    # Load persisted claim tokens so pairing QRs survive restarts.
    _claims_load()
    # Restore per-session model switches so a server restart doesn't
    # silently reset every chat/voice session back to the default model.
    _load_session_overrides()
    # Load the persisted models cache so the app's model picker is instant
    # after a server restart, then warm it in the background.
    _models_load_from_disk()
    asyncio.get_running_loop().run_in_executor(None, _refresh_models_background, "")


# ─── Main ────────────────────────────────────────────────────────────────


def _open_browser_best_effort(url: str) -> None:
    """Open the pairing page in the default browser (best-effort; silent on
    headless machines / Termux where xdg-open doesn't exist)."""
    try:
        import platform as _platform
        sysname = _platform.system().lower()
        if "linux" in sysname:
            subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif "darwin" in sysname:
            subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif "windows" in sysname:
            subprocess.Popen(["cmd", "/c", "start", "", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


if __name__ == "__main__":
    # ─── OmniRoute Auto Lifecycle Management ───
    # Bridge auto-detects, starts (if needed), and configures OmniRoute
    # so the user always gets free auto-routing models without manual setup.
    from hermes_features import _read_config_yaml as _rcfg

    def _ensure_omnirouter() -> tuple[str, str, str]:
        """
        Full OmniRoute lifecycle: install → start → configure.
        Returns (base_url, api_key, model_name).
        """
        or_url = "http://localhost:20128/v1"
        or_key = ""
        or_model = "auto/best-coding"
        or_bin = None

        # ─── 1. Install if missing ───
        try:
            or_bin = shutil.which("omniroute")
        except Exception:
            or_bin = None

        if not or_bin:
            logger.info("OmniRoute not found — installing via npm...")
            try:
                subprocess.run(
                    ["npm", "install", "-g", "omniroute"],
                    capture_output=True, timeout=120,
                )
                try:
                    or_bin = shutil.which("omniroute")
                except Exception:
                    or_bin = None
                if or_bin:
                    logger.info("OmniRoute installed successfully: %s", or_bin)
                else:
                    logger.warning("OmniRoute install completed but binary not found")
            except Exception as e:
                logger.warning("OmniRoute install failed: %s — continuing with defaults", e)
        else:
            logger.info("OmniRoute already installed: %s", or_bin)

        # ─── 2. Ensure server is running ───
        running = False
        if or_bin:
            try:
                req = urllib.request.Request(f"{or_url}/models", headers={"User-Agent": "HermesBridge/2.0"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    if resp.status == 200:
                        running = True
                        logger.info("OmniRoute server already running on %s", or_url)
            except Exception:
                pass

            if not running:
                logger.info("Starting OmniRoute server...")
                try:
                    proc = subprocess.Popen(
                        [or_bin, "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    logger.info("OmniRoute started (PID %d)", proc.pid)
                    # Wait up to 20s for readiness
                    for attempt in range(20):
                        time.sleep(1)
                        try:
                            req = urllib.request.Request(f"{or_url}/models", headers={"User-Agent": "HermesBridge/2.0"})
                            with urllib.request.urlopen(req, timeout=2) as resp:
                                if resp.status == 200:
                                    running = True
                                    logger.info("OmniRoute ready after %ds", attempt + 1)
                                    break
                        except Exception:
                            continue
                    if not running:
                        logger.warning("OmniRoute did not become ready within 20s")
                except Exception as e:
                    logger.warning("Failed to start OmniRoute: %s", e)

        # ─── 3. Setup API key ───
        # Read from env first (set by OmniRoute's .env or Hermes config)
        or_key = os.environ.get("HERMES_CUSTOM_LOCALHOST_20128_API_KEY", "")
        if not or_key:
            or_key = os.environ.get("OMNIROUTE_API_KEY", "")
        if not or_key and running:
            or_key = f"sk-{secrets.token_hex(16)}"
            os.environ["HERMES_CUSTOM_LOCALHOST_20128_API_KEY"] = or_key
            logger.info("Generated new OmniRoute API key")
        if or_key:
            logger.info("OmniRoute API key: %s...%s", or_key[:8], or_key[-4:])

        # ─── 4. Apply config.yaml override (if present) ───
        try:
            _cfg = _rcfg()
            _cp_list = _cfg.get("custom_providers", {})
            if isinstance(_cp_list, dict):
                _cp_list = [_cp_list]
            for _cp in _cp_list:
                if isinstance(_cp, dict) and "omnirouter" in (_cp.get("name", "") or "").lower():
                    _cu = _cp.get("base_url", "").rstrip("/")
                    if _cu:
                        or_url = _cu
                    _ke = _cp.get("key_env", "")
                    if _ke and os.environ.get(_ke):
                        or_key = os.environ.get(_ke, "")
                    _ms = _cp.get("models", [])
                    if _ms:
                        or_model = _ms[0] if isinstance(_ms, list) else _ms
                    break
        except Exception:
            pass

        return or_url, or_key, or_model

    # Run OmniRoute setup
    _or_url, _or_key, _or_model = _ensure_omnirouter()
    if _or_url:
        AI_BASE_URL = _or_url
    if _or_key:
        AI_API_KEY = _or_key
    if _or_model:
        AI_MODEL = _or_model

    logger.info("AI provider: %s @ %s (key: %s)", AI_MODEL, AI_BASE_URL, "set" if AI_API_KEY else "none")

    # Initialize command handler with current config
    from hermes_features import set_config
    set_config(AI_MODEL, AI_BASE_URL)
    print(f"🤖 Hermes Mobile Bridge v2 starting on http://{HOST}:{PORT}")
    print(f"   AI: {AI_MODEL} @ {AI_BASE_URL}")
    print(f"   Store: {STORE_PATH}")
    try:
        _setup_ip = _detect_host_ip()
        print(f"📱 Pairing page: http://{_setup_ip}:{PORT}/setup?token={SETUP_TOKEN}")
        print(f"📱 App setup: http://{_setup_ip}:{PORT}/setup/connect?token={SETUP_TOKEN}")
        print(f"📱 Scan QR:   http://{_setup_ip}:{PORT}/setup/qr?token={SETUP_TOKEN}")
        # Auto-open the secure pairing page so the QR is one click away.
        _open_browser_best_effort(f"http://{_setup_ip}:{PORT}/setup?token={SETUP_TOKEN}")
    except Exception:
        pass
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=False)