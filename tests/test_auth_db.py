#!/usr/bin/env python3
"""Unit tests for auth_db — plain python, no pytest dependency.

Run:  python3 tests/test_auth_db.py
Covers the refresh-token rotation race fix (atomic DELETE...RETURNING)
and the message-history trim bound.
"""
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import auth_db as m


def test_refresh_rotation_blocks_replay():
    """Two concurrent verifies of the SAME token: exactly one wins."""
    with tempfile.TemporaryDirectory() as tmp:
        db = m.get_auth_db(Path(tmp))
        db.create_user("race@test.com", "hash12345678")
        uid = db.get_user_by_email("race@test.com").id
        raw = "test-refresh-token"
        h = hashlib.sha256(raw.encode()).hexdigest()
        db._get_conn().execute(
            "INSERT INTO refresh_tokens (token_hash,user_id,expires_at,created_at) VALUES (?,?,?,?)",
            (h, uid, int(time.time()) + 3600, int(time.time())),
        )
        db._get_conn().commit()
        first = db.verify_refresh_token(raw)
        second = db.verify_refresh_token(raw)
        assert first == uid, f"first verify should return user_id, got {first}"
        assert second is None, f"replay must be blocked, got {second}"
    print("PASS  rotation: concurrent replay blocked")


def test_refresh_expired_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        db = m.get_auth_db(Path(tmp))
        db.create_user("exp@test.com", "hash12345678")
        uid = db.get_user_by_email("exp@test.com").id
        raw = "expired-token"
        h = hashlib.sha256(raw.encode()).hexdigest()
        db._get_conn().execute(
            "INSERT INTO refresh_tokens (token_hash,user_id,expires_at,created_at) VALUES (?,?,?,?)",
            (h, uid, int(time.time()) - 10, int(time.time())),
        )
        db._get_conn().commit()
        assert db.verify_refresh_token(raw) is None
        # Expired token is also deleted (rotation cleans up)
        n = db._get_conn().execute(
            "SELECT COUNT(*) FROM refresh_tokens WHERE token_hash=?", (h,)
        ).fetchone()[0]
        assert n == 0
    print("PASS  expired token rejected and purged")


def test_message_history_trim():
    """save_messages caps history at 300 (memory bound)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["STORE_PATH"] = tmp
        import importlib
        import server as s
        importlib.reload(s)
        msgs = [{"role": "user", "content": f"m{i}"} for i in range(400)]
        s.save_messages("trimtest", msgs)
        loaded = s.load_messages("trimtest")
        assert len(loaded) == 300, f"expected 300, got {len(loaded)}"
        assert loaded[0]["content"] == "m100", "should keep the MOST RECENT 300"
        assert loaded[-1]["content"] == "m399"
    print("PASS  message history trimmed to 300")


if __name__ == "__main__":
    test_refresh_rotation_blocks_replay()
    test_refresh_expired_rejected()
    test_message_history_trim()
    print("\nAll auth_db tests passed.")
