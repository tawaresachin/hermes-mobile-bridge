"""True-disconnect smoke test against the LIVE server.

Client opens a stream, reads ONE chunk, then closes the connection.
The detached generation must still complete and save the response —
even though the SSE generator was cancelled by the disconnect.
"""
import json
import os
import sys
import time
import uuid

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import httpx

BASE = "http://localhost:9119"
EMAIL = f"disconnect-live-{int(time.time())}@gmail.com"

with httpx.Client(timeout=10) as c:
    reg = c.post(f"{BASE}/auth/register", json={"email": EMAIL, "password": "testpass123"})
    token = (reg.json() or {}).get("access_token") or (reg.json() or {}).get("token")
    assert token, f"register failed: {reg.status_code} {reg.text[:200]}"
    auth = {"Authorization": f"Bearer {token}"}
    print("authed")

    sid = f"live-disconnect-{uuid.uuid4().hex[:8]}"
    # True disconnect: read one chunk, then leave the context (closes conn)
    with httpx.stream("POST", f"{BASE}/api/chat/stream",
                      json={"query": "hi", "session_id": sid},
                      headers=auth, timeout=30) as r:
        assert r.status_code == 200, r.status_code
        first = next(r.iter_bytes())
        print("got first chunk:", first[:50])

# Connection CLOSED. Detached task must finish + save.
import os as _os
from pathlib import Path
store = Path(_os.path.expanduser("~/.hermes-mobile-server"))
deadline = time.time() + 30
content = None
while time.time() < deadline:
    p = store / f"messages_{sid}.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        for m in reversed(data):
            if m.get("role") == "assistant" and (m.get("content") or "").strip():
                content = m["content"]
                break
    if content:
        break
    time.sleep(0.5)

assert content, "FAIL: assistant response NOT saved after true disconnect"
print("PASS: full response saved after true client disconnect:", content[:80])
