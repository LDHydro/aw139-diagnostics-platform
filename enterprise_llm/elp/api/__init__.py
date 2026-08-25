"""HTTP API routers."""

from . import (
    v1_admin,
    v1_ask,
    v1_auth,
    v1_dev,
    v1_documents,
    v1_latex,
    v1_maintenance,
    v1_mel,
    v1_openai,
)

ROUTERS = [
    v1_auth.router,
    v1_ask.router,
    v1_openai.router,
    v1_documents.router,
    v1_maintenance.router,
    v1_mel.router,
    v1_latex.router,
    v1_dev.router,
    v1_admin.router,
]

__all__ = ["ROUTERS"]
