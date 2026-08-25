"""
Central configuration for the Enterprise LLM Platform (ELP).

Everything is driven by environment variables prefixed with ``ELP_`` (see
``.env.example``).  Nested settings use a double underscore, e.g.::

    ELP_AUTH__OIDC_ISSUER=https://login.microsoftonline.com/<tenant>/v2.0
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------
# Sub-configurations
# ----------------------------------------------------------------------

class InferenceSettings(BaseModel):
    """Local vLLM / TEI endpoints running on the RTX 5090."""

    # Primary generation model (chat, document Q&A, LaTeX, code).
    chat_base_url: str = "http://127.0.0.1:8101/v1"
    chat_model: str = "Qwen/Qwen3-32B-AWQ"
    chat_api_key: str = "local"

    # Optional dedicated code model.  When empty the chat model is used.
    code_base_url: str = ""
    code_model: str = ""

    # Embeddings (bge-m3 => 1024 dims).
    embed_base_url: str = "http://127.0.0.1:8102/v1"
    embed_model: str = "BAAI/bge-m3"
    embed_dim: int = 1024
    embed_batch_size: int = 16

    # Cross-encoder reranker (bge-reranker-v2-m3).
    rerank_url: str = "http://127.0.0.1:8103/rerank"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_enabled: bool = True

    # Generation defaults.
    max_context_tokens: int = 32768
    max_output_tokens: int = 2048
    default_temperature: float = 0.2
    request_timeout_s: float = 180.0
    connect_timeout_s: float = 10.0


class AuthSettings(BaseModel):
    """SSO / Active Directory access control."""

    # "oidc" for Entra ID / ADFS / Keycloak, "ldap" for direct AD bind,
    # "disabled" only for isolated development.
    mode: Literal["oidc", "ldap", "disabled"] = "oidc"

    # --- OIDC (Microsoft Entra ID, ADFS 2019+, Keycloak) ---
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = ""
    # Explicit JWKS override; normally discovered from the issuer.
    oidc_jwks_url: str = ""
    jwks_cache_seconds: int = 3600
    # Claim that carries AD group membership.  Entra ID emits "groups"
    # (object IDs) or "roles" (app roles); ADFS can be configured to emit
    # group SAM names.
    groups_claim: str = "groups"
    # Fall back to app roles when Entra ID truncates the groups claim
    # (the "groups overage" case, >200 groups).
    roles_claim: str = "roles"
    allow_group_overage_fallback: bool = True
    # Clock skew tolerance for token validation.
    leeway_seconds: int = 60

    # --- LDAP direct bind (no IdP available) ---
    ldap_url: str = ""
    ldap_bind_dn: str = ""
    ldap_bind_password: str = ""
    ldap_user_base_dn: str = ""
    ldap_user_filter: str = "(sAMAccountName={username})"
    ldap_group_attribute: str = "memberOf"
    ldap_use_ssl: bool = True

    # --- Group -> role mapping (JSON object in env, or config/groups.yaml) ---
    # e.g. {"AW139-Maintenance-Admins": "admin", "AW139-Engineering": "engineer"}
    group_role_map: dict[str, str] = Field(default_factory=dict)
    # Groups permitted to reach the platform at all.  Empty = any
    # authenticated user (still subject to per-document ACLs).
    allowed_groups: list[str] = Field(default_factory=list)
    default_role: str = "reader"

    # --- Service accounts for in-house applications ---
    api_key_header: str = "X-API-Key"
    api_key_prefix: str = "elp_"


class RagSettings(BaseModel):
    """Retrieval configuration for the governing documents."""

    chunk_target_tokens: int = 700
    chunk_overlap_tokens: int = 90
    chunk_max_tokens: int = 1200
    # Retrieval fan-out before reranking.
    vector_top_k: int = 40
    keyword_top_k: int = 40
    # Reciprocal-rank-fusion constant.
    rrf_k: int = 60
    # Passages handed to the model after reranking.
    final_top_k: int = 8
    min_rerank_score: float = 0.05
    # Refuse to answer when nothing clears this bar.
    min_answer_confidence: float = 0.12
    max_context_chars: int = 48000
    # Platform admins do NOT bypass document ACLs by default: being able
    # to administer the system is not the same as being cleared to read
    # every department's governing documents.
    admin_bypass_acl: bool = False


class FederationSettings(BaseModel):
    """Talking to other internal LLM/AI systems."""

    enabled: bool = True
    registry_path: str = str(BASE_DIR / "config" / "peers.yaml")
    max_parallel: int = 4
    per_peer_timeout_s: float = 45.0
    # Consult peers automatically when the router selects them.
    auto_consult: bool = True
    # Hard ceiling on peers consulted for a single question.
    max_peers_per_query: int = 3


class MaintenanceSettings(BaseModel):
    """Predictive maintenance scheduling."""

    # Days of utilisation history used to project the daily flight-hour rate.
    utilization_window_days: int = 90
    # Exponential smoothing factor for the utilisation forecast.
    utilization_alpha: float = 0.25
    # Fallback rate when an aircraft has no usable history.
    default_daily_flight_hours: float = 2.0
    default_daily_cycles: float = 3.0
    # Planning horizon for forecasts and the event planner.
    forecast_horizon_days: int = 365
    planning_horizon_days: int = 120
    # Tasks whose due windows fall within this many days are bundled into
    # one maintenance event to minimise aircraft downtime.
    bundling_window_days: int = 14
    # Labour capacity used by the planner.
    shift_hours_per_day: float = 8.0
    technicians_per_shift: int = 4
    # Warn when remaining margin drops below this fraction of the interval.
    warning_threshold_pct: float = 0.10
    # Never ground more than this many aircraft on the same day.
    max_concurrent_aircraft_down: int = 2
    # Refuse to bundle a task if doing it this early throws away more
    # than this fraction of its interval.
    max_interval_waste_pct: float = 0.25


class LatexSettings(BaseModel):
    enabled: bool = True
    # "tectonic" (self-contained, recommended) or "latexmk".
    engine: Literal["tectonic", "latexmk"] = "tectonic"
    compile_timeout_s: int = 120
    max_source_bytes: int = 2_000_000
    artifact_dir: str = str(BASE_DIR / "artifacts" / "latex")
    template_dir: str = str(BASE_DIR / "elp" / "latex" / "templates")


class ReportSettings(BaseModel):
    """
    Operational reporting against NAMIS.

    Every limit here exists because a generated query runs unattended on a
    schedule: the blast radius of a bad one is not one slow page, it is a
    production database at 03:00 with nobody watching.
    """

    enabled: bool = True

    # --- NAMIS connection -------------------------------------------
    # "sql" for a direct read-only database connection, "rest" for an API.
    namis_kind: Literal["sql", "rest", "disabled"] = "disabled"
    # SQLAlchemy URL. MUST point at a READ-ONLY database account.
    namis_dsn: str = ""
    namis_rest_base_url: str = ""
    # Name of the environment variable holding the credential, never the
    # credential itself.
    namis_auth_env_var: str = "NAMIS_PASSWORD"
    namis_auth_header: str = "Authorization"
    namis_auth_type: Literal["none", "bearer", "api_key"] = "bearer"

    # --- Query safety ------------------------------------------------
    # Schemas and tables the reporting account may read. Empty means "any
    # table the database account can see", which is only acceptable when
    # that account is itself tightly scoped.
    allowed_schemas: list[str] = Field(default_factory=list)
    allowed_tables: list[str] = Field(default_factory=list)
    # Columns never returned, matched case-insensitively on the column name.
    redacted_columns: list[str] = Field(
        default_factory=lambda: ["password", "passwd", "secret", "token", "ssn", "cpf"]
    )
    max_rows: int = 5000
    statement_timeout_ms: int = 30000
    # Rows handed to the model when it narrates the result. The full set
    # still reaches the report; only the narration sample is capped.
    narration_row_sample: int = 50
    max_cell_chars: int = 500

    # --- Scheduling ---------------------------------------------------
    # A generated query may not run unattended until a person approves it.
    require_approval_for_schedule: bool = True
    max_concurrent_runs: int = 2
    run_retention_days: int = 365
    artifact_dir: str = str(BASE_DIR / "artifacts" / "reports")


class DevSettings(BaseModel):
    """Code-assist / application development."""

    enabled: bool = True
    # Absolute paths the code assistant may read.  Nothing outside these
    # roots is ever opened.
    workspace_roots: list[str] = Field(default_factory=list)
    max_file_bytes: int = 400_000
    max_files_per_request: int = 40
    excluded_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", "target", ".mypy_cache", ".pytest_cache",
        ]
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ELP_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Enterprise LLM Platform"
    environment: Literal["development", "staging", "production"] = "production"
    host: str = "0.0.0.0"
    port: int = 8080
    log_level: str = "INFO"
    # Origins allowed to call the API from a browser (in-house web apps).
    cors_origins: list[str] = Field(default_factory=list)

    database_url: str = "postgresql+asyncpg://elp:elp@127.0.0.1:5432/elp"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_echo: bool = False

    inference: InferenceSettings = Field(default_factory=InferenceSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    rag: RagSettings = Field(default_factory=RagSettings)
    federation: FederationSettings = Field(default_factory=FederationSettings)
    maintenance: MaintenanceSettings = Field(default_factory=MaintenanceSettings)
    latex: LatexSettings = Field(default_factory=LatexSettings)
    reports: ReportSettings = Field(default_factory=ReportSettings)
    dev: DevSettings = Field(default_factory=DevSettings)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v: Any) -> Any:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
