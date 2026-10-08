"""Vercel entrypoint: Vercel serves the FastAPI `app` exported from index.py.

Vercel's filesystem is read-only except /tmp. Without DATABASE_URL the SQLite
database and the development encryption key live in /tmp, which is private to
each running copy of the function and can reset at any time: fine for a quick
demo, not for real use. For real use connect a Postgres database (Vercel
Storage -> Neon sets DATABASE_URL) and set RESPONSE_ENCRYPTION_KEY and
ADMIN_API_KEY in the Vercel project's environment variables.
"""
import dataclasses
import logging
import os
from pathlib import Path

ON_VERCEL = bool(os.environ.get("VERCEL"))
if ON_VERCEL: os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/assessments.db")

import config  # noqa: E402

if ON_VERCEL: config.settings = dataclasses.replace(config.settings, dev_key_path=Path("/tmp/.dev_encryption_key"))

SHARED_DB = not config.settings.database_url.startswith("sqlite")
if ON_VERCEL and SHARED_DB and not config.settings.response_encryption_key: logging.getLogger("assessment").error("RESPONSE_ENCRYPTION_KEY is not set: every function instance will use its own temporary key, so stored answers and saved AI keys cannot be read back reliably. Set it in Vercel > Settings > Environment Variables.")

from api_server import app  # noqa: E402,F401
