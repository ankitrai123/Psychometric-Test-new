"""Vercel entrypoint: Vercel serves the FastAPI `app` exported from index.py.

Vercel's filesystem is read-only except /tmp, so on Vercel the SQLite database
and the development encryption key default to /tmp. That storage is temporary
(it can reset at any time), which is fine for testing. For real use set
DATABASE_URL (e.g. a Postgres URL), RESPONSE_ENCRYPTION_KEY and ADMIN_API_KEY
in the Vercel project's environment variables.
"""
import dataclasses
import os
from pathlib import Path

ON_VERCEL = bool(os.environ.get("VERCEL"))
if ON_VERCEL: os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/assessments.db")

import config  # noqa: E402

if ON_VERCEL: config.settings = dataclasses.replace(config.settings, dev_key_path=Path("/tmp/.dev_encryption_key"))

from api_server import app  # noqa: E402,F401
