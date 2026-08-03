#!/usr/bin/env python3
"""
Authentication database layer — SQLite for users, tokens, and sessions.
Single file, zero external deps beyond stdlib + bcrypt + jwt.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bcrypt
import jwt

# ─── Constants ───

DB_FILENAME = "auth.db"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 168  # 7 days
REFRESH_EXPIRE_DAYS = 30
BCRYPT_ROUNDS = 12

# ─── Data classes ───


@dataclass
class User:
    id: int
    email: str
    password_hash: bytes
    created_at: int


@dataclass
class RefreshToken:
    token_hash: str  # SHA-256 of the raw token
    user_id: int
    expires_at: int
    created_at: int


# ─── Core DB ───


class AuthDB:
    """Thread-safe wrapper around SQLite for auth data."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS refresh_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_refresh_user ON refresh_tokens(user_id);
            CREATE INDEX IF NOT EXISTS idx_refresh_expires ON refresh_tokens(expires_at);
            """
        )
        conn.commit()
        self._conn = conn

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._init_db()
        return self._conn

    # ─── User operations ───

    def create_user(self, email: str, password: str) -> User:
        """Register a new user. Raises ValueError if email exists."""
        conn = self._get_conn()
        email = email.strip().lower()
        if not email or "@" not in email:
            raise ValueError("Invalid email format")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        # Hash password
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(BCRYPT_ROUNDS))

        try:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (email, password_hash, int(time.time())),
            )
            conn.commit()
            user_id = cur.lastrowid
            return User(
                id=user_id,
                email=email,
                password_hash=password_hash,
                created_at=int(time.time()),
            )
        except sqlite3.IntegrityError:
            raise ValueError("Email already registered")

    def get_user_by_email(self, email: str) -> Optional[User]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, email, password_hash, created_at FROM users WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
        if row:
            return User(id=row[0], email=row[1], password_hash=row[2], created_at=row[3])
        return None

    def verify_password(self, password: str, password_hash: bytes) -> bool:
        try:
            return bcrypt.checkpw(password.encode(), password_hash)
        except Exception:
            return False

    # ─── Refresh token operations ───

    @staticmethod
    def _hash_token(token: str) -> str:
        """SHA-256 of the raw refresh token (we never store plaintext)."""
        return hashlib.sha256(token.encode()).hexdigest()

    def create_refresh_token(self, user_id: int) -> tuple[str, int]:
        """Generate a new refresh token. Returns (raw_token, expires_at)."""
        conn = self._get_conn()
        raw_token = secrets.token_urlsafe(48)  # 64 bytes → 86 chars
        token_hash = self._hash_token(raw_token)
        now = int(time.time())
        expires_at = now + REFRESH_EXPIRE_DAYS * 86400

        conn.execute(
            "INSERT INTO refresh_tokens (token_hash, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (token_hash, user_id, expires_at, now),
        )
        conn.commit()
        return raw_token, expires_at

    def verify_refresh_token(self, raw_token: str) -> Optional[int]:
        """Verify a refresh token. Returns user_id if valid, None otherwise.
        Implements rotation atomically: DELETE ... RETURNING in ONE statement
        closes the replay race — two concurrent requests with the same token
        cannot both pass (the second sees zero rows)."""
        conn = self._get_conn()
        token_hash = self._hash_token(raw_token)
        now = int(time.time())

        # Purge expired tokens for this hash (dead tokens — no race concern).
        conn.execute(
            "DELETE FROM refresh_tokens WHERE token_hash = ? AND expires_at < ?",
            (token_hash, now),
        )
        # Atomic rotation: delete-and-return in ONE statement closes the
        # replay race — two concurrent requests with the same token cannot
        # both pass (the second sees zero rows).
        row = conn.execute(
            "DELETE FROM refresh_tokens WHERE token_hash = ? AND expires_at >= ? "
            "RETURNING user_id",
            (token_hash, now),
        ).fetchone()
        conn.commit()

        if not row:
            return None
        return row[0]

    def revoke_all_refresh_tokens(self, user_id: int) -> None:
        """Log out user from all devices."""
        conn = self._get_conn()
        conn.execute("DELETE FROM refresh_tokens WHERE user_id = ?", (user_id,))
        conn.commit()

    def cleanup_expired_tokens(self) -> int:
        """Remove expired refresh tokens. Returns count deleted."""
        conn = self._get_conn()
        now = int(time.time())
        cur = conn.execute(
            "DELETE FROM refresh_tokens WHERE expires_at < ?", (now,)
        )
        conn.commit()
        return cur.rowcount

    # ─── JWT operations ───

    @staticmethod
    def create_access_token(user_id: int, email: str, secret: str) -> str:
        now = int(time.time())
        payload = {
            "sub": str(user_id),
            "email": email,
            "iat": now,
            "exp": now + JWT_EXPIRE_HOURS * 3600,
            "type": "access",
        }
        return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)

    @staticmethod
    def decode_access_token(token: str, secret: str) -> Optional[dict]:
        try:
            return jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None


# ─── Factory (singleton per path) ───

_auth_db_cache: dict[str, AuthDB] = {}

def get_auth_db(store_path: Path) -> AuthDB:
    """Get or create the AuthDB instance (singleton per store_path)."""
    key = str(store_path.absolute())
    if key not in _auth_db_cache:
        db_path = store_path / DB_FILENAME
        _auth_db_cache[key] = AuthDB(db_path)
    return _auth_db_cache[key]


if __name__ == "__main__":
    # Quick self-test
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        db = get_auth_db(Path(tmp))
        u = db.create_user("test@example.com", "password123")
        print(f"Created user: {u.id} {u.email}")

        # Verify password
        assert db.verify_password("password123", u.password_hash)
        assert not db.verify_password("wrong", u.password_hash)

        # Refresh tokens
        token, exp = db.create_refresh_token(u.id)
        assert db.verify_refresh_token(token) == u.id
        # Second use should fail (rotated)
        assert db.verify_refresh_token(token) is None

        # New token works
        token2, _ = db.create_refresh_token(u.id)
        assert db.verify_refresh_token(token2) == u.id

        # JWT
        secret = "test-secret"
        jwt_token = db.create_access_token(u.id, u.email, secret)
        payload = db.decode_access_token(jwt_token, secret)
        assert payload["sub"] == str(u.id)
        assert payload["email"] == u.email

        print("All self-tests passed!")