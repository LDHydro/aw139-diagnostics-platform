"""Shared fixtures.  These tests exercise pure logic - no database, no GPU."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Settings are read at import time, so they must be in place before any
# platform module is imported.
os.environ.setdefault("ELP_ENVIRONMENT", "development")
os.environ.setdefault("ELP_AUTH__MODE", "disabled")
os.environ.setdefault("ELP_DATABASE_URL", "postgresql+asyncpg://elp:elp@127.0.0.1:5432/elp_test")

import pytest  # noqa: E402


@pytest.fixture
def maintenance_settings():
    from elp.config import MaintenanceSettings

    return MaintenanceSettings()


@pytest.fixture
def rag_settings():
    from elp.config import RagSettings

    return RagSettings()
