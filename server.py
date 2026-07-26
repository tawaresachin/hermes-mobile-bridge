#!/usr/bin/env python3
"""
Hermes Mobile Bridge Server
REST/SSE backend that the Hermes Mobile Android app connects to.
Forwards chat to OpenCode Zen (or any OpenAI-compatible API).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ─── Config ────────────────────────────────────────────────────────────

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9119"))
API_KEY = os.getenv("HERMES_API_KEY", "hermes123")  # mobile app auth key
STORE_PATH = Path(os.getenv("STORE_PATH", os.path.expanduser("~/.hermes-mobile-server")))

# OpenCode Zen (or any OpenAI-compatible provider)
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://opencode.ai/zen/v1")
AI_API_KEY = os.getenv("AI_API_KEY") or os.getenv("OPENCODE_ZEN_API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "deepseek-v4-flash-free")

STORE_PATH.mkdir(parents=True, exist_ok=True)

# ─── Data store ────────────────────────────────────────────────────────


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


# ─── App ───────────────────────────────────────────────────────────────

app = FastAPI(title="Hermes Mobile Bridge", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_auth(request: Request) -> None:
    """Check Bearer token against configured API key."""
    auth = request.headers.get("Authorization", "")
    expected = f"Bearer {API_KEY}"
    if auth != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ─── Models ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    query: str
    session_id: str | None = None
    stream: bool = True


# ─── Endpoints ─────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/sessions")
async def list_sessions(request: Request):
    verify_auth(request)
    return load_sessions()


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    verify_auth(request)
    session_id = body.session_id or uuid.uuid4().hex[:8]

    # Save user message & create/update session
    msgs = load_messages(session_id)
    msgs.append({"role": "user", "content": body.query, "timestamp": time.time()})
    save_messages(session_id, msgs)

    sessions = load_sessions()
    existing = next((s for s in sessions if s["id"] == session_id), None)
    if existing:
        existing["messageCount"] = (existing.get("messageCount", 0) or 0) + 1
        existing["updatedAt"] = int(time.time() * 1000)
    else:
        sessions.insert(0, {
            "id": session_id,
            "title": body.query[:60] + ("…" if len(body.query) > 60 else ""),
            "messageCount": 1,
            "createdAt": int(time.time() * 1000),
            "updatedAt": int(time.time() * 1000),
        })
    save_sessions(sessions)

    async def event_generator() -> AsyncGenerator[str, None]:
        full_response = ""
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                openai_messages = [
                    {"role": "system", "content": "You are Hermes, a helpful AI assistant. Be concise and accurate."},
                ]
                # Add conversation history (up to last 20)
                for m in msgs[-20:]:
                    openai_messages.append({"role": m["role"], "content": m["content"]})

                payload = {
                    "model": AI_MODEL,
                    "messages": openai_messages,
                    "stream": True,
                }
                headers = {}
                if AI_API_KEY:
                    headers["Authorization"] = f"Bearer {AI_API_KEY}"
                headers["Content-Type"] = "application/json"

                async with client.stream(
                    "POST",
                    f"{AI_BASE_URL}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        yield f"data: {json.dumps({'content': f'⚠️ API error {resp.status_code}: {error_text.decode()[:200]}'})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response += content
                                yield f"data: {json.dumps({'content': content})}\n\n"
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            yield f"data: {json.dumps({'content': f'⚠️ Connection error: {e}'})}\n\n"

        # Save assistant response
        if full_response:
            msgs.append({"role": "assistant", "content": full_response, "timestamp": time.time()})
            save_messages(session_id, msgs)

        yield "data: [DONE]\n\n"

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
async def chat_sync(body: ChatRequest, request: Request):
    verify_auth(request)
    session_id = body.session_id or uuid.uuid4().hex[:8]

    msgs = load_messages(session_id)
    msgs.append({"role": "user", "content": body.query, "timestamp": time.time()})

    sessions = load_sessions()
    existing = next((s for s in sessions if s["id"] == session_id), None)
    if existing:
        existing["messageCount"] = (existing.get("messageCount", 0) or 0) + 1
        existing["updatedAt"] = int(time.time() * 1000)
    else:
        sessions.insert(0, {
            "id": session_id,
            "title": body.query[:60] + ("…" if len(body.query) > 60 else ""),
            "messageCount": 1,
            "createdAt": int(time.time() * 1000),
            "updatedAt": int(time.time() * 1000),
        })
    save_sessions(sessions)

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            openai_messages = [
                {"role": "system", "content": "You are Hermes, a helpful AI assistant. Be concise and accurate."},
            ]
            for m in msgs[-20:]:
                openai_messages.append({"role": m["role"], "content": m["content"]})

            payload = {
                "model": AI_MODEL,
                "messages": openai_messages,
                "stream": False,
            }
            headers = {}
            if AI_API_KEY:
                headers["Authorization"] = f"Bearer {AI_API_KEY}"
            headers["Content-Type"] = "application/json"
            resp = await client.post(
                f"{AI_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                msgs.append({"role": "assistant", "content": content, "timestamp": time.time()})
                save_messages(session_id, msgs)
                return {"response": content, "session_id": session_id}
            else:
                return {"response": f"⚠️ API error: {resp.status_code}", "session_id": session_id}
    except Exception as e:
        return {"response": f"⚠️ Error: {e}", "session_id": session_id}


# ─── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"🤖 Hermes Mobile Bridge starting on http://{HOST}:{PORT}")
    print(f"   API Key: {API_KEY}")
    print(f"   AI: {AI_MODEL} @ {AI_BASE_URL}")
    print(f"   Store: {STORE_PATH}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
