"""Admin dashboard sign-in: username + password -> signed session token.

Tokens are stateless (HMAC-SHA256 over a small JSON payload), so they work
across serverless instances without a session table. The signing key is
derived from the admin password: changing ADMIN_PASSWORD signs everyone out.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time

from config import Settings


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _key(cfg: Settings) -> bytes:
    password = cfg.effective_admin_password or ""
    return hashlib.sha256(b"admin-session-v1:" + password.encode()).digest()


def check_credentials(cfg: Settings, username: str, password: str) -> bool:
    """Constant-time check; both comparisons always run."""
    expected = cfg.effective_admin_password
    if not expected:
        return False
    user_ok = secrets.compare_digest(username.strip().lower().encode(), cfg.admin_username.lower().encode())
    pass_ok = secrets.compare_digest(password.encode(), expected.encode())
    return user_ok and pass_ok


def issue_token(cfg: Settings, now: float | None = None) -> tuple[str, int]:
    expires = int((now or time.time()) + cfg.admin_session_hours * 3600)
    payload = _b64(json.dumps({"u": cfg.admin_username, "exp": expires}, separators=(",", ":")).encode())
    signature = _b64(hmac.new(_key(cfg), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{signature}", expires


def verify_token(cfg: Settings, token: str, now: float | None = None) -> str | None:
    """Return the username for a valid, unexpired token, else None."""
    if not cfg.admin_auth_enabled or token.count(".") != 1:
        return None
    payload, signature = token.split(".")
    expected = _b64(hmac.new(_key(cfg), payload.encode(), hashlib.sha256).digest())
    if not secrets.compare_digest(signature.encode(), expected.encode()):
        return None
    try:
        data = json.loads(_unb64(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or int(data.get("exp", 0)) <= (now or time.time()):
        return None
    return str(data.get("u", ""))
