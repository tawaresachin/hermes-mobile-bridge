#!/usr/bin/env python3
"""
JWT + Password utilities for Hermes Bridge.
Re-exports auth_db functions for convenience.
"""
from __future__ import annotations

import os
import secrets
import hashlib
import time
from pathlib import Path
from typing import Optional

import bcrypt
import jwt

from auth_db import AuthDB, get_auth_db, DB_FILENAME

# ─── Constants ───

BCRYPT_ROUNDS = 12
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 168  # 7 days
REFRESH_EXPIRE_DAYS = 30

JWT_SECRET_FILENAME = ".jwt_secret"


# ─── Password ───


def hash_password(password: str) -> bytes:
    """Return bcrypt hash of password."""
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(BCRYPT_ROUNDS))


def verify_password(password: str, password_hash: bytes) -> bool:
    """Check password against bcrypt hash. Safe on bad hashes."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash)
    except Exception:
        return False


# ─── JWT ───


def get_jwt_secret(store_path: Path) -> str:
    """Load or generate JWT secret key (32+ bytes)."""
    secret_path = store_path / JWT_SECRET_FILENAME
    if secret_path.exists():
        return secret_path.read_text().strip()
    # Generate new 32-byte secret
    secret = secrets.token_urlsafe(32)  # 43 chars
    secret_path.write_text(secret)
    secret_path.chmod(0o600)
    return secret


def create_access_token(user_id: int, email: str, secret: str) -> str:
    """Create a short-lived JWT access token."""
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + JWT_EXPIRE_HOURS * 3600,
        "type": "access",
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, secret: str) -> Optional[dict]:
    """Decode and validate JWT. Returns payload dict or None."""
    try:
        return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ─── Refresh Token ───


def generate_refresh_token() -> tuple[str, str]:
    """Generate a cryptographically random refresh token.
    Returns (raw_token, sha256_hex_hash).
    """
    raw = secrets.token_urlsafe(48)  # 64 bytes → ~86 chars
    hashed = hashlib.sha256(raw.encode()).hexdigest()
    return raw, hashed


def verify_refresh_token_hash(raw_token: str, stored_hash: str) -> bool:
    """Constant-time compare of refresh token hash."""
    computed = hashlib.sha256(raw_token.encode()).hexdigest()
    return secrets.compare_digest(computed, stored_hash)


# ─── Convenience: full auth flow helpers ───


async def register_user(store_path: Path, email: str, password: str) -> dict:
    """Full registration flow. Returns token dict."""
    auth_db = get_auth_db(store_path)
    user = auth_db.create_user(email, password)
    secret = get_jwt_secret(store_path)
    access = create_access_token(user.id, user.email, secret)
    refresh_raw, _ = generate_refresh_token()
    refresh_hash = hashlib.sha256(refresh_raw.encode()).hexdigest()
    # Store refresh token
    auth_db._get_conn().execute(
        "INSERT INTO refresh_tokens (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (
            refresh_hash,
            user.id,
            int(time.time()) + REFRESH_EXPIRE_DAYS * 86400,
            int(time.time()),
        ),
    )
    auth_db._get_conn().commit()
    return {
        "token": access,
        "refresh_token": refresh_raw,
        "user_id": str(user.id),
        "email": user.email,
    }


async def login_user(store_path: Path, email: str, password: str) -> Optional[dict]:
    """Full login flow. Returns token dict or None if invalid."""
    auth_db = get_auth_db(store_path)
    user = auth_db.get_user_by_email(email)
    if not user or not verify_password(password, user.password_hash):
        return None
    secret = get_jwt_secret(store_path)
    access = create_access_token(user.id, user.email, secret)
    refresh_raw, _ = generate_refresh_token()
    refresh_hash = hashlib.sha256(refresh_raw.encode()).hexdigest()
    auth_db._get_conn().execute(
        "INSERT INTO refresh_tokens (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (
            refresh_hash,
            user.id,
            int(time.time()) + REFRESH_EXPIRE_DAYS * 86400,
            int(time.time()),
        ),
    )
    auth_db._get_conn().commit()
    return {
        "token": access,
        "refresh_token": refresh_raw,
        "user_id": str(user.id),
        "email": user.email,
    }


async def refresh_tokens(store_path: Path, refresh_token: str) -> Optional[dict]:
    """Rotate refresh token. Returns new token pair or None if invalid/expired."""
    auth_db = get_auth_db(store_path)
    # Verify and consume old token
    user_id = auth_db.verify_refresh_token(refresh_token)
    if not user_id:
        return None
    # Get user email
    row = auth_db._get_conn().execute(
        "SELECT email FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    if not row:
        return None
    email = row[0]
    # Issue new tokens
    secret = get_jwt_secret(store_path)
    access = create_access_token(user_id, email, secret)
    refresh_raw, _ = generate_refresh_token()
    refresh_hash = hashlib.sha256(refresh_raw.encode()).hexdigest()
    auth_db._get_conn().execute(
        "INSERT INTO refresh_tokens (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (
            refresh_hash,
            user_id,
            int(time.time()) + REFRESH_EXPIRE_DAYS * 86400,
            int(time.time()),
        ),
    )
    auth_db._get_conn().commit()
    return {
        "token": access,
        "refresh_token": refresh_raw,
        "user_id": str(user_id),
        "email": email,
    }


# ─── Self-test ───

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp)
        # Test password
        pw = "testpassword123"
        h = hash_password(pw)
        assert verify_password(pw, h)
        assert not verify_password("wrong", h)
        print("Password hashing OK")

        # Test JWT
        secret = get_jwt_secret(store)
        assert len(secret) >= 32
        tok = create_access_token(1, "test@test.com", secret)
        payload = decode_access_token(tok, secret)
        assert payload["sub"] == "1"
        assert payload["email"] == "test@test.com"
        assert decode_access_token("invalid", secret) is None
        print("JWT OK")

        # Test refresh token
        raw, hashed = generate_refresh_token()
        assert verify_refresh_token_hash(raw, hashed)
        assert not verify_refresh_token_hash("wrong", hashed)
        print("Refresh token OK")

        # Test full flows
        async def test_flows():
            reg = await register_user(store, "test@test.com", "password123")
            assert reg["token"]
            assert reg["refresh_token"]
            print("Register OK")

            login = await login_user(store, "test@test.com", "password123")
            assert login and login["token"]
            print("Login OK")

            refreshed = await refresh_tokens(store, login["refresh_token"])
            assert refreshed and refreshed["refresh_token"] != login["refresh_token"]
            # Old token should be consumed
            assert await refresh_tokens(store, login["refresh_token"]) is None
            print("Refresh rotation OK")

        import asyncio

        asyncio.run(test_flows())

    print("All self-tests passed!")