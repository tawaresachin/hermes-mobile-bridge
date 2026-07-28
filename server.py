#!/usr/bin/env python3
"""
Hermes Mobile Bridge Server
REST/SSE backend that the Hermes Mobile Android app connects to.
Forwards chat to OpenCode Zen (or any OpenAI-compatible API).

New in v2: User authentication with JWT + refresh tokens.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
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

# ─── Config ──────────────────────────────────────────────────────────────

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9119"))

# AI provider (OpenAI-compatible)
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://opencode.ai/zen/v1")
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("OPENCODE_ZEN_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "deepseek-v4-flash-free")

# Optional static API key for backward compatibility
# If set, accepts Authorization: Bearer <this_key> as fallback
HERMES_API_KEY = os.getenv("HERMES_API_KEY", "")

# Data storage
STORE_PATH = Path(os.getenv("STORE_PATH", os.path.expanduser("~/.hermes-mobile-server")))
STORE_PATH.mkdir(parents=True, exist_ok=True)

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


def _save_user_message(session_id: str, query: str) -> None:
    """Append the user's query to the message history and persist."""
    msgs = load_messages(session_id)
    msgs.append({"role": "user", "content": query, "timestamp": time.time()})
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


def _build_openai_messages(session_id: str) -> list[dict]:
    """Build the conversation history array for the OpenAI-compatible API."""
    msgs = load_messages(session_id)
    return [
        {"role": "system", "content": "You are Hermes, a helpful AI assistant. Be concise and accurate."},
        *[{"role": m["role"], "content": m["content"]} for m in msgs[-20:]],
    ]


# ─── Models ──────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    stream: bool = True


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Auth dependency ────────────────────────────────────────────────────


async def verify_bearer(request: Request) -> dict:
    """
    FastAPI dependency: validate Authorization: Bearer ***
    Returns the decoded JWT payload (contains sub=user_id, email).
    Raises 401 on failure.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth[7:].strip()
    payload = decode_access_token(token, JWT_SECRET)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    # Verify user still exists
    user_id = int(payload["sub"])
    user = auth_db.get_user_by_email(payload["email"])
    if not user or user.id != user_id:
        raise HTTPException(status_code=401, detail="User not found")

    return payload


# Public endpoints that don't require auth
PUBLIC_PATHS = {"/health", "/diag", "/auth/register", "/auth/login", "/auth/refresh"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Skip auth for public paths; enforce for everything else."""
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


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, user: dict = Depends(verify_bearer)):
    """Streaming chat endpoint with full agent tool execution."""
    session_id = body.session_id or uuid.uuid4().hex[:8]

    _save_user_message(session_id, body.query)
    _upsert_session(session_id, body.query)

    # Load conversation history
    openai_messages = _build_openai_messages(session_id)

    # Use the agent loop for full tool execution
    from agent_loop import AgentLoop

    loop = AgentLoop(
        ai_base_url=AI_BASE_URL,
        ai_api_key=AI_API_KEY,
        ai_model=AI_MODEL,
    )

    async def event_generator() -> AsyncGenerator[str, None]:
        full_assistant_response = ""
        try:
            async for event in loop.run(openai_messages, body.query):
                yield event
                # Collect text for saving
                if '"type":"text"' in event and '"content":"' in event:
                    try:
                        data = json.loads(event[6:])  # Strip "data: "
                        if data.get("type") == "text":
                            full_assistant_response += data.get("content", "")
                    except (json.JSONDecodeError, IndexError):
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': f'⚠ Agent error: {e}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Save assistant response to history
        if full_assistant_response:
            _save_assistant_message(session_id, full_assistant_response)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat_sync(body: ChatRequest, user: dict = Depends(verify_bearer)):
    """Non-streaming chat endpoint (for simple clients)."""
    session_id = body.session_id or uuid.uuid4().hex[:8]

    _save_user_message(session_id, body.query)
    _upsert_session(session_id, body.query)

    openai_messages = _build_openai_messages(session_id)

    try:
        headers = {"Content-Type": "application/json"}
        if AI_API_KEY:
            headers["Authorization"] = f"Bearer {AI_API_KEY}"

        payload = {
            "model": AI_MODEL,
            "messages": openai_messages,
            "stream": False,
        }

        resp = await _http_client.post(
            f"{AI_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            _save_assistant_message(session_id, content)
            return {"response": content, "session_id": session_id}
        else:
            return {"response": f"⚠️ API error: {resp.status_code}", "session_id": session_id}
    except Exception as e:
        return {"response": f"⚠️ Error: {e}", "session_id": session_id}


# ─── Discovery / Registry Integration ──────────────────────────────

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
            import re
            match = re.search(r'https://[a-z0-9.-]+\.trycloudflare\.com', content)
            if match:
                tunnel_url = match.group(0)

    if tunnel_url:
        await client.heartbeat(
            tunnel_url=tunnel_url,
            platform=__import__("sys").platform,
            version="2.0.0",
        )
        print(f"   Tunnel URL: {tunnel_url}")
    else:
        print("⚠ No tunnel URL detected — set HERMES_TUNNEL_URL or start cloudflared")

    # Start periodic heartbeat
    import asyncio

    async def heartbeat_loop():
        while True:
            await asyncio.sleep(180)  # every 3 minutes
            try:
                url = os.getenv("HERMES_TUNNEL_URL", tunnel_url or "")
                if url:
                    await client.heartbeat(url, platform=__import__("sys").platform)
            except Exception:
                pass

    asyncio.create_task(heartbeat_loop())


@app.on_event("startup")
async def on_startup():
    await start_discovery()


# ─── Main ────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print(f"🤖 Hermes Mobile Bridge v2 starting on http://{HOST}:{PORT}")
    print(f"   AI: {AI_MODEL} @ {AI_BASE_URL}")
    print(f"   Store: {STORE_PATH}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")