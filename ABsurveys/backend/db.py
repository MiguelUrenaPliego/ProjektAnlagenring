"""
=======================================================================
db.py — MongoDB connection and collection accessors
=======================================================================

Reads MONGODB_URI from the environment (or falls back to localhost).
All collections are accessed through this module so the client is
created exactly once per process.

Collections
-----------
image_state   — one document per (scenario, img_id); holds TrueSkill
                state columns (score_*, uncertainty_*, n_answers_*,
                active_batch_*) plus static metadata (path, img_type,
                batch_id, incompatible_ids, _serve_path).

user_answers  — one document per submitted answer, append-only log.
=======================================================================
"""

from __future__ import annotations

import os
from motor.motor_asyncio import AsyncIOMotorClient

_MONGODB_URI: str = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
_DB_NAME: str = os.environ.get("MONGODB_DB", "survey_app")

_client: AsyncIOMotorClient | None = None


def _get_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(_MONGODB_URI)
    return _client


def get_db():
    return _get_client()[_DB_NAME]


def image_state_col():
    return get_db()["image_state"]


def user_answers_col():
    return get_db()["user_answers"]