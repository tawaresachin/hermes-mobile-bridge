"""Smoke test: client disconnects mid-stream -> detached generation must
still complete and save the response server-side."""
import asyncio
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import server
import agent_loop
from fastapi.testclient import TestClient

SESSION = f"disconnect-test-{int(time.time())}"

# Fake slow agent: yields a few events with pauses, then finishes.
async def fake_run(self, *args, **kwargs):
    for i in range(5):
        await asyncio.sleep(0.15)
        yield f"data: {json.dumps({'type': 'text', 'content': f'chunk-{i} '})}\n\n"

# AgentLoop is imported lazily inside the handler from agent_loop — patch
# the class method there.
agent_loop.AgentLoop.run = fake_run  # type: ignore

# Authenticate for real — the auth dependency is captured at decoration time.
msgs_path = server._messages_path(SESSION)
if msgs_path.exists():
    msgs_path.unlink()  # clean slate BEFORE the stream

with TestClient(server.app) as client:
    reg = client.post("/auth/register", json={
        "email": f"disconnect-test-{int(time.time())}@gmail.com",
        "password": "testpass123",
    })
    token = reg.json().get("access_token") or reg.json().get("token")
    if not token:
        # Maybe registration returns a claim/QR flow — try login path fallback
        token = None
    assert token, f"no token from register: {reg.status_code} {reg.text[:200]}"
    auth = {"Authorization": f"Bearer {token}"}
    print("authed")

    with client.stream("POST", "/api/chat/stream",
                       json={"query": "hello", "session_id": SESSION},
                       headers=auth) as resp:
        assert resp.status_code == 200, resp.status_code
        # Read only the first event, then ABORT the connection
        it = resp.iter_bytes()
        first = next(it)
        print("got first event:", first[:60])

# Connection closed mid-stream. Wait for the detached task to finish + save.
deadline = time.time() + 15
saved = False
while time.time() < deadline:
    if msgs_path.exists():
        data = json.loads(msgs_path.read_text(encoding="utf-8"))
        if any(m.get("role") == "assistant" for m in data):
            saved = True
            break
    time.sleep(0.5)

assert saved, "FAIL: assistant response was NOT saved after disconnect"
data = json.loads(msgs_path.read_text(encoding="utf-8"))
content = [m["content"] for m in data if m.get("role") == "assistant"][0]
assert content == "chunk-0 chunk-1 chunk-2 chunk-3 chunk-4 ", repr(content)
print("PASS: full response saved after client disconnect:", content)
