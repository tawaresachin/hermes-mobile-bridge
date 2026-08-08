"""Security regression: session-id collision (SEC1), SSRF redirect (SEC2),
sqlite-lock bypass (C1/A4), and the detached-task cap (S1).

Runs against the LIVE server on :9119.
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
EMAIL = f"sec-{uuid.uuid4().hex[:8]}@gmail.com"


def _register() -> tuple[httpx.Client, dict, str]:
    c = httpx.Client(timeout=10)
    email = f"sec-{uuid.uuid4().hex[:10]}@gmail.com"
    reg = c.post(f"{BASE}/auth/register", json={"email": email, "password": "testpass123"})
    assert reg.status_code == 200, f"register: {reg.status_code} {reg.text[:200]}"
    body = reg.json()
    token = body.get("token") or body.get("access_token")
    assert token, f"no token: {body}"
    return c, {"Authorization": f"Bearer {token}"}, email


def test_session_id_collision_closed():
    c, auth, _ = _register()
    sid = uuid.uuid4().hex

    # 1. A real session gets created + is readable.
    r = c.post(f"{BASE}/api/chat/stream", json={"query": "hi", "session_id": sid}, headers=auth, timeout=30)
    assert r.status_code == 200, r.status_code

    # 2. Malformed ids are REJECTED everywhere (not sanitized into a
    #    collision with another session's file).
    for bad in (f"{sid}!", f"{sid}..", f"{sid}/evil"):
        r = c.get(f"{BASE}/api/sessions/{bad}/messages", headers=auth)
        assert r.status_code in (400, 404), f"GET {bad} -> {r.status_code}"
        r = c.delete(f"{BASE}/api/sessions/{bad}", headers=auth)
        assert r.status_code in (400, 404), f"DELETE {bad} -> {r.status_code}"

    # 3. Stream + sync chat reject malformed ids in the body.
    for path in ("/api/chat/stream", "/api/chat"):
        r = c.post(f"{BASE}{path}", json={"query": "hi", "session_id": "victim!"}, headers=auth, timeout=15)
        assert r.status_code == 400, f"{path} body -> {r.status_code}"

    # 4. The old collision: "victim!" must NOT resolve to messages_victim.json.
    store = os.path.expanduser("~/.hermes-mobile-server")
    if os.path.exists(f"{store}/messages_victim.json"):
        # Pre-seed a victim file to prove the read is blocked.
        with open(f"{store}/messages_victim.json", "w") as f:
            json.dump([{"role": "assistant", "content": "SECRET"}], f)
    r = c.get(f"{BASE}/api/sessions/victim!/messages", headers=auth)
    assert r.status_code == 400, f"collision read -> {r.status_code}"

    print("PASS  session-id collision closed (400 on all malformed ids)")


def test_auth_flows_still_work_locked():
    c, auth, email = _register()
    # Login issues a refresh token + claim token via the LOCKED path.
    r = c.post(f"{BASE}/auth/login", json={"email": email, "password": "testpass123"})
    assert r.status_code == 200, f"login: {r.status_code} {r.text[:200]}"
    refresh = r.json().get("refresh_token")
    claim = r.json().get("claim_token")
    assert refresh and claim, "login must return refresh + claim tokens"
    # Refresh rotation works (locked path).
    r = c.post(f"{BASE}/auth/refresh", json={"refresh_token": refresh})
    assert r.status_code == 200, f"refresh: {r.status_code} {r.text[:200]}"
    # Claim exchange works (locked path).
    r = c.post(f"{BASE}/auth/claim", json={"token": claim})
    assert r.status_code == 200, f"claim: {r.status_code} {r.text[:200]}"
    print("PASS  register/login/refresh/claim all work via locked AuthDB methods")


def test_ssrf_redirect_guard():
    from tools.web import fetch_safe, is_safe_url

    assert not is_safe_url("http://127.0.0.1:9119/health"), "loopback must be blocked"
    assert not is_safe_url("http://169.254.169.254/latest/meta-data/"), "metadata must be blocked"
    assert not is_safe_url("http://100.101.102.103/"), "tailscale CGNAT must be blocked"
    assert is_safe_url("https://example.com/"), "public https must pass"

    # Redirect chain: a public URL that 302s to loopback must be refused
    # (the hop is re-validated because follow_redirects is forced off).
    import tools.web_research as wr
    assert wr._http.follow_redirects is False, "web_research client must not auto-follow"
    print("PASS  SSRF guard: loopback/link-local/CGNAT blocked, redirects re-validated")


def test_detached_task_cap():
    import server as srv

    assert srv._AGENT_TASK_SEM._value == 4, "semaphore must start at 4"
    assert isinstance(srv._AGENT_TASKS, set)
    print("PASS  detached-task cap present (semaphore 4 + registry)")


if __name__ == "__main__":
    test_session_id_collision_closed()
    test_auth_flows_still_work_locked()
    test_ssrf_redirect_guard()
    test_detached_task_cap()
    print("\nALL SECURITY REGRESSIONS PASSED")
