#!/usr/bin/env python3
"""
Hermes Mobile Bridge Server
REST/SSE backend that the Hermes Mobile Android app connects to.
Forwards chat to OpenCode Zen (or any OpenAI-compatible API).

New in v2: User authentication with JWT + refresh tokens.
"""
from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, Depends, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr

# Local auth modules
from auth_db import get_auth_db, AuthDB
from auth_handler import (
    hash_password,
    verify_password,
    get_jwt_secret,
    create_access_token,
    decode_access_token,
    register_user,
    login_user,
    refresh_tokens,
    REFRESH_EXPIRE_DAYS,
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
            logger.info("Loaded .env: %s=%s...", _key, _val[:8] + "..." if len(_val) > 8 else _val)
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


def _resolve_model(session_id: str) -> str:
    """Get the effective model for a session: override if set, else default."""
    return _session_model_overrides.get(session_id, AI_MODEL)


def _resolve_base_url(session_id: str) -> str:
    """Get the effective base URL for a session."""
    return _session_base_url_overrides.get(session_id, AI_BASE_URL)


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

        if is_global:
            # Write to .env as new default (persistent)
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
                env_path.write_text(content, encoding="utf-8")
            # Also update the module-level default for immediate effect
            import server as _srv_mod
            _srv_mod.AI_MODEL = model_name
            return True, f"✅ Model permanently switched to `{model_name}` (global)."
        else:
            # Session-scoped override (live, no restart)
            _session_model_overrides[session_id] = model_name
            # Also switch base URL if we know it for this model
            if model_name in _model_base_url_map:
                _session_base_url_overrides[session_id] = _model_base_url_map[model_name]
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

# Shared HTTP client — one connection pool for all requests
_http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

# Initialize auth DB
auth_db = get_auth_db(STORE_PATH)
JWT_SECRET = get_jwt_secret(STORE_PATH)

# ─── Data store (sessions & messages) ────────────────────────────────────


def _sessions_path() -> Path:
    return STORE_PATH / "sessions.json"


def _messages_path(session_id: str) -> Path:
    return STORE_PATH / f"messages_{session_id}.json"


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
    _messages_path(session_id).write_text(json.dumps(msgs, indent=2, default=str))


# ─── Shared helpers ─────────────────────────────────────────────────────


def _save_user_message(session_id: str, query: str, attachment_url: str = "", attachment_type: str = "") -> None:
    """Append the user's query to the message history and persist."""
    msgs = load_messages(session_id)
    entry = {"role": "user", "content": query, "timestamp": time.time()}
    if attachment_url:
        entry["attachment_url"] = attachment_url
        entry["attachment_type"] = attachment_type
    msgs.append(entry)
    save_messages(session_id, msgs)


def _save_assistant_message(session_id: str, content: str) -> None:
    """Append the assistant response to the message history and persist."""
    msgs = load_messages(session_id)
    msgs.append({"role": "assistant", "content": content, "timestamp": time.time()})
    save_messages(session_id, msgs)


def _upsert_session(session_id: str, query: str) -> None:
    """Create or update a session entry from a user query."""
    sessions = load_sessions()
    existing = next((s for s in sessions if s["id"] == session_id), None)
    if existing:
        existing["messageCount"] = (existing.get("messageCount", 0) or 0) + 1
        existing["updatedAt"] = int(time.time() * 1000)
    else:
        sessions.insert(
            0,
            {
                "id": session_id,
                "title": query[:60] + ("…" if len(query) > 60 else ""),
                "messageCount": 1,
                "createdAt": int(time.time() * 1000),
                "updatedAt": int(time.time() * 1000),
            },
        )
    save_sessions(sessions)


def _read_attachment_content(attach_url: str, attach_type: str) -> str:
    """Try to read uploaded file content from the local filesystem.
    Returns a string snippet to embed in the message, or empty string if unreadable."""
    if not attach_url or not attach_url.startswith("/uploads/"):
        return ""

    try:
        # Parse /uploads/{session_id}/{filename}
        parts = attach_url.split("/")
        if len(parts) < 4:
            return ""
        sess = parts[2]
        fname = "/".join(parts[3:])
        fpath = UPLOADS_DIR / sess / fname
        # Path-traversal guard: resolved path MUST stay inside the uploads dir
        resolved = fpath.resolve()
        if not str(resolved).startswith(str(UPLOADS_DIR.resolve())) or ".." in parts[2:]:
            return ""
        if not resolved.exists() or not resolved.is_file():
            return ""

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
                        return f"\n\n--- Attached image (OCR extracted): {fname} ---\n{ocr_text}\n--- End of OCR text ---"
                except Exception:
                    pass
            
            return f"[{mime_hint} file, {size} bytes — content not readable as text]"

        # Truncate to reasonable length
        if len(text) > 6000:
            text = text[:6000] + "\n… [truncated]"

        return f"\n\n--- Attached file: {fname} ---\n{text}\n--- End of attached file ---"

    except Exception as e:
        logger.warning("Failed to read attachment %s: %s", attach_url, e)
        return ""


def _build_openai_messages(session_id: str) -> list[dict]:
    """Build the conversation history array for the OpenAI-compatible API.
    Reads uploaded file content from local storage and embeds it as text.
    Does NOT include a system prompt — AgentLoop adds its own."""
    msgs = load_messages(session_id)
    result = []
    for m in msgs[-20:]:
        content = m["content"]
        attach_url = m.get("attachment_url", "")
        attach_type = m.get("attachment_type", "")

        if attach_url:
            file_snippet = _read_attachment_content(attach_url, attach_type)
            enhanced = content
            if file_snippet:
                enhanced += file_snippet
            else:
                label = "Image" if attach_type and "image" in attach_type else "File"
                enhanced += f"\n\n[{label} attached: {attach_url}]"
            result.append({"role": m["role"], "content": enhanced})
        else:
            result.append({"role": m["role"], "content": content})
    return result


# ─── Models ──────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: bool = True
    attachment_url: Optional[str] = None
    attachment_type: Optional[str] = None


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


# ─── App ─────────────────────────────────────────────────────────────────


app = FastAPI(title="Hermes Mobile Bridge", version="2.0.0")

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

    # Fallback: check static API key (backward compat)
    if HERMES_API_KEY and token == HERMES_API_KEY:
        return {"sub": "0", "email": "legacy@api-key", "type": "legacy"}

    raise HTTPException(status_code=401, detail="Invalid or expired token")


# Public endpoints that don't require auth
PUBLIC_PATHS = {"/health", "/diag", "/auth/register", "/auth/login", "/auth/refresh", "/setup/qr", "/setup/connect"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Skip auth for public paths; enforce for everything else."""
    if request.url.path in PUBLIC_PATHS or request.url.path.startswith(("/auth/", "/uploads/")):
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
        return await call_next(request)
    
    # Fallback: check if it's the static HERMES_API_KEY (backward compat)
    if HERMES_API_KEY and token == HERMES_API_KEY:
        request.state.user = {"sub": "0", "email": "legacy@api-key", "type": "legacy"}
        return await call_next(request)
    
    # Neither worked
    return Response(status_code=401, content='{"detail":"Invalid or expired token"}', media_type="application/json")


@app.post("/auth/register")
async def auth_register(body: RegisterRequest):
    """Register a new user. Returns access + refresh tokens."""
    try:
        result = await register_user(STORE_PATH, body.email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.post("/auth/login")
async def auth_login(body: LoginRequest):
    """Log in with email + password. Returns access + refresh tokens."""
    result = await login_user(STORE_PATH, body.email, body.password)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return result


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
    real_ip = request.headers.get("X-Forwarded-For", request.client.host if request.client else "unknown")
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
        },
    }


@app.get("/api/sessions")
async def list_sessions(user: dict = Depends(verify_bearer)):
    """List all chat sessions for the authenticated user."""
    return load_sessions()


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(verify_bearer)):
    """Delete a session and its messages."""
    sessions = load_sessions()
    sessions = [s for s in sessions if s.get("id") != session_id]
    save_sessions(sessions)
    msgs_path = _messages_path(session_id)
    if msgs_path.exists():
        msgs_path.unlink()
    logger.info(f"Deleted session {session_id} for user {user['sub']}")
    return {"status": "ok", "deleted": session_id}


@app.get("/api/models")
async def list_models(session_id: str = "", user: dict = Depends(verify_bearer)):
    """List available models — only from providers we can actually query."""
    from hermes_features import _list_available_models, AI_MODEL, AI_BASE_URL, _read_config_yaml
    import urllib.request, urllib.error

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

    # 2. Query OpenCode/Zen models (no auth needed)
    try:
        req = urllib.request.Request(
            "https://opencode.ai/zen/v1/models",
            headers={"User-Agent": "HermesMobileBridge/2.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            oc_data = json.loads(resp.read())
        for m in oc_data.get("data", []):
            mid = m.get("id", "")
            if mid and mid not in seen:
                seen.add(mid)
                is_free = mid.endswith("-free")
                models.append({
                    "id": mid,
                    "name": mid,
                    "isFree": is_free,
                    "isVision": any(k in mid.lower() for k in ("vl", "vision", "multimodal")),
                    "provider": "opencode",
                    "baseUrl": "https://opencode.ai/zen/v1",
                })
    except Exception as e:
        logger.warning("OpenCode/Zen query failed: %s", e)

    # 4. Try OpenRouter models if key is configured
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {or_key}"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                or_data = json.loads(resp.read())
            for m in or_data.get("data", []):
                mid = m.get("id", "")
                if mid and mid not in seen:
                    seen.add(mid)
                    is_free = mid.endswith(":free") or mid.endswith("-free")
                    models.append({
                        "id": mid,
                        "name": mid.split("/")[-1] if "/" in mid else mid,
                        "isFree": is_free,
                        "isVision": any(k in mid.lower() for k in ("vl", "vision", "multimodal")),
                        "provider": "openrouter",
                        "baseUrl": "https://openrouter.ai/api/v1",
                    })
        except Exception as e:
            logger.warning("OpenRouter query failed: %s", e)

    # 5. Custom providers from config.yaml
    config = _read_config_yaml()
    raw_cp = config.get("custom_providers", {})
    cp_list = raw_cp if isinstance(raw_cp, list) else [raw_cp]
    for cp in cp_list:
        if not isinstance(cp, dict):
            continue
        cp_model = cp.get("model", "")
        cp_name = cp.get("name", "custom")
        if cp_model and cp_model not in seen:
            seen.add(cp_model)
            models.append({
                "id": cp_model,
                "name": cp_model.split("/")[-1] if "/" in cp_model else cp_model,
                "isFree": False,
                "isVision": False,
                "provider": cp_name.lower().replace(" ", "-"),
                "baseUrl": cp.get("base_url", ""),
            })

    # Build base URL map for provider-aware switching
    for m in models:
        if m.get("baseUrl"):
            _model_base_url_map[m["id"]] = m["baseUrl"]

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

    _save_user_message(session_id, query, body.attachment_url or "", body.attachment_type or "")
    _upsert_session(session_id, query)

    # ─── Pre-check for special commands (model switch, etc.) ───
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

    # Load conversation history
    openai_messages = _build_openai_messages(session_id)

    # Use session-scoped model override if available
    effective_model = _resolve_model(session_id)

    # Use the agent loop for full tool execution
    from agent_loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
    from hermes_features import set_config
    effective_base_url = _resolve_base_url(session_id)
    set_config(effective_model, effective_base_url)

    loop = AgentLoop(
        ai_base_url=effective_base_url,
        ai_api_key=AI_API_KEY,
        ai_model=effective_model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        full_assistant_response = ""
        try:
            async for event in loop.run(openai_messages, query, session_id=session_id,
                                         attachment_url=body.attachment_url or "",
                                         attachment_type=body.attachment_type or ""):
                yield event
                # Collect text for saving
                if event.startswith('data: {'):
                    try:
                        raw = event[6:].strip()
                        data = json.loads(raw)
                        if data.get('type') == 'text':
                            full_assistant_response += data.get('content', '')
                    except (json.JSONDecodeError, IndexError):
                        pass
        except asyncio.CancelledError:
            # Client disconnected — save what we have
            logger.warning(f"Stream cancelled for session {session_id}, saving partial response")
            if full_assistant_response:
                _save_assistant_message(session_id, full_assistant_response)
            return
        except Exception as e:
            logger.error(f"Stream error for session {session_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': f'⚠ Agent error: {e}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Save assistant response to history
        if full_assistant_response:
            logger.info(f"Saving assistant response for session {session_id} ({len(full_assistant_response)} chars)")
            _save_assistant_message(session_id, full_assistant_response)
        else:
            logger.warning(f"Empty full_assistant_response for session {session_id}")

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

    _save_user_message(session_id, body.query)
    _upsert_session(session_id, body.query)

    # ─── Pre-check for special commands (model switch, etc.) ───
    handled, cmd_response = _handle_special_command(body.query, session_id)
    if handled and cmd_response:
        return {"response": cmd_response, "session_id": session_id}

    # Use session-scoped model override if available
    effective_model = _resolve_model(session_id)
    effective_base_url = _resolve_base_url(session_id)

    # Use agent loop but just collect the final response
    from agent_loop import AgentLoop, DEFAULT_SYSTEM_PROMPT
    from hermes_features import set_config as _set_config
    _set_config(effective_model, effective_base_url)

    loop = AgentLoop(
        ai_base_url=effective_base_url,
        ai_api_key=AI_API_KEY,
        ai_model=effective_model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )

    openai_messages = _build_openai_messages(session_id)
    final_response = ""
    error_msg = None

    try:
        async for event in loop.run(openai_messages, body.query):
            # Try to parse every SSE data event (cover both text and error types)
            if event.startswith('data: {'):
                try:
                    raw = event[6:].strip()
                    data = json.loads(raw)
                    if data.get('type') == 'text':
                        final_response += data.get('content', '')
                    elif data.get('type') == 'error':
                        error_msg = data.get('content', 'Agent error')
                except (json.JSONDecodeError, IndexError):
                    pass
    except Exception as e:
        error_msg = str(e)

    if error_msg:
        return {"response": f"⚠️ {error_msg}", "session_id": session_id}

    if final_response:
        _save_assistant_message(session_id, final_response)
    else:
        final_response = f"⚠️ No response generated"

    return {"response": final_response, "session_id": session_id}


# ─── File Upload Endpoint ──────────────────────────────────────────────


@app.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
):
    """Upload a file to the session's uploads directory."""
    # Verify JWT auth — header only (query-param tokens leak into access logs)
    payload = None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:].strip()
        payload = decode_access_token(token, JWT_SECRET)
    if not payload:
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    # Validate user still exists
    user_id = int(payload["sub"])
    user = auth_db.get_user_by_email(payload["email"])
    if not user or user.id != user_id:
        raise HTTPException(status_code=401, detail="User not found")

    # Determine session_id from query param, or use user-specific default
    session_id = request.query_params.get("session_id", payload.get("sub", "default"))

    # Path-traversal guard: session_id must be a plain identifier (no separators)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id or ""):
        raise HTTPException(status_code=400, detail="Invalid session_id")

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

    # Save the file
    content = await file.read()
    dest_path.write_bytes(content)

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(safe_filename)
    mime_type = mime_type or file.content_type or "application/octet-stream"

    logger.info(
        "Uploaded file %s (%d bytes, %s) for session %s",
        safe_filename,
        len(content),
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


@app.get("/setup/qr")
async def setup_qr(token: str = ""):
    """Return a QR code PNG that the mobile app scans to auto-configure."""
    if token != SETUP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid setup token")
    import io
    try:
        import qrcode
        from qrcode.image.pil import PilImage
    except ImportError:
        raise HTTPException(status_code=501, detail="qrcode[pil] not installed — run: pip install qrcode[pil]")

    # Tailscale IP (primary) → tunnel URL (fallback) → LAN IP (last resort)
    ts_ip = _detect_host_ip()
    if ts_ip and ts_ip.startswith("100."):
        setup_url = f"hermes://connect?host={ts_ip}&port={PORT}&key={HERMES_API_KEY}&setup={SETUP_TOKEN}"
    else:
        tunnel_url = _detect_tunnel_url()
        if tunnel_url:
            setup_url = f"hermes://connect?url={tunnel_url}&key={HERMES_API_KEY}&setup={SETUP_TOKEN}"
        else:
            host_ip = ts_ip or "127.0.0.1"
            setup_url = f"hermes://connect?host={host_ip}&port={PORT}&key={HERMES_API_KEY}&setup={SETUP_TOKEN}"

    img = qrcode.make(setup_url, image_factory=PilImage)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/setup/connect")
async def setup_connect(token: str = ""):
    """Return connection JSON for the app to auto-configure."""
    if token != SETUP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid setup token")
    ts_ip = _detect_host_ip()
    if ts_ip and ts_ip.startswith("100."):
        return {
            "host": ts_ip,
            "port": PORT,
            "url": f"http://{ts_ip}:{PORT}",
            "key": HERMES_API_KEY,
            "model": AI_MODEL,
            "provider": AI_BASE_URL,
            "version": "2.2.1",
        }
    tunnel_url = _detect_tunnel_url()
    if tunnel_url:
        return {
            "url": tunnel_url,
            "key": HERMES_API_KEY,
            "model": AI_MODEL,
            "provider": AI_BASE_URL,
            "version": "2.2.1",
        }
    host_ip = _detect_host_ip()
    return {
        "host": host_ip,
        "port": PORT,
        "url": f"http://{host_ip}:{PORT}",
        "key": HERMES_API_KEY,
        "model": AI_MODEL,
        "provider": AI_BASE_URL,
        "version": "2.2.1",
    }


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
    """Detect the best IP for the app to connect to."""
    # 0. Env override (set by plugin's Tailscale auto-setup)
    ts_ip = os.environ.get("HERMES_TAILSCALE_IP", "")
    if ts_ip and ts_ip.startswith("100."):
        return ts_ip

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
            version="2.0.0",
        )
        print(f"   Tunnel URL: {tunnel_url}")
    else:
        print("⚠ No tunnel URL detected — set HERMES_TUNNEL_URL or start cloudflared")

    # Start periodic heartbeat
    async def heartbeat_loop():
        while True:
            await asyncio.sleep(180)
            try:
                url = os.getenv("HERMES_TUNNEL_URL", tunnel_url or "")
                if url:
                    await client.heartbeat(url, platform=sys.platform)
            except Exception:
                pass

    asyncio.create_task(heartbeat_loop())


@app.on_event("startup")
async def on_startup():
    await start_discovery()


# ─── Main ────────────────────────────────────────────────────────────────


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
        print(f"📱 App setup: http://{_setup_ip}:{PORT}/setup/connect?token={SETUP_TOKEN}")
        print(f"📱 Scan QR:   http://{_setup_ip}:{PORT}/setup/qr?token={SETUP_TOKEN}")
    except Exception:
        pass
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")