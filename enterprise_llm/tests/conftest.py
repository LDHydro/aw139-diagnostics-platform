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

import json  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def maintenance_settings():
    from elp.config import MaintenanceSettings

    return MaintenanceSettings()


@pytest.fixture
def rag_settings():
    from elp.config import RagSettings

    return RagSettings()


def _catalog_module():
    from elp.reports import catalog as module

    return module


@pytest.fixture
def catalog_file(tmp_path):
    """A miniature catalogue in the shape the report generator exports."""
    data = {
        "groups": [
            {"name": "Work Requests", "tables": ["WorkRequest", "WRStatusHist"]},
            {"name": "Aircraft & Assets", "tables": ["AIRCRAFT", "ASSETLCF"]},
        ],
        "tables": {
            "WorkRequest": {
                "name": "WorkRequest", "table": "WorkRequest", "schema": "dbo",
                "database": "NAMISNNSS", "group": "Work Requests",
                "objectType": "TABLE", "rowCount": 120000,
                "columns": [
                    {"name": "WRId", "sql": "uniqueidentifier", "kind": "text",
                     "nullable": False, "pk": True},
                    {"name": "WRNo", "sql": "char(7)", "kind": "text", "nullable": True},
                    {"name": "StatusCd", "sql": "char(3)", "kind": "text", "nullable": True},
                    {"name": "AssetKey", "sql": "decimal(9,0)", "kind": "number"},
                    {"name": "AssetSite", "sql": "varchar(15)", "kind": "text"},
                ],
            },
            "WRStatusHist": {
                "name": "WRStatusHist", "schema": "dbo", "database": "NAMISNNSS",
                "columns": [
                    {"name": "WRId", "sql": "uniqueidentifier", "pk": True},
                    {"name": "StatusCdAfter", "sql": "char(3)"},
                ],
            },
            "AIRCRAFT": {
                "name": "AIRCRAFT", "schema": "dbo", "database": "NAMISNNSS",
                "group": "Aircraft & Assets",
                "columns": [
                    {"name": "AssetKey", "sql": "decimal(9,0)", "pk": True},
                    {"name": "AssetSite", "sql": "varchar(15)", "pk": True},
                    {"name": "TailNumber", "sql": "varchar(15)"},
                ],
            },
            "FlightRecordHeaders": {
                "name": "FlightRecordHeaders", "schema": "dbo",
                "database": "AMO_NASAWeb",
                "columns": [{"name": "SortieID", "sql": "varchar(40)"}],
            },
        },
        "relationships": [
            {
                "left": "AIRCRAFT", "right": "WorkRequest",
                "on": [
                    {"leftColumn": "AssetKey", "rightColumn": "AssetKey"},
                    {"leftColumn": "AssetSite", "rightColumn": "AssetSite"},
                ],
                "database": "NAMISNNSS", "confidence": "fk",
            },
            {
                "left": "WorkRequest", "right": "WRStatusHist",
                "on": [{"leftColumn": "WRId", "rightColumn": "WRId"}],
                "confidence": "fk",
            },
        ],
        "lookups": {"aircraftTail": {}},
    }
    path = tmp_path / "namis-catalog.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def catalog(catalog_file):
    module = _catalog_module()
    module.reset_catalog()
    loaded = module.load_catalog(catalog_file)
    yield loaded
    module.reset_catalog()
