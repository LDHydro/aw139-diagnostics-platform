"""Request and response models for the public API."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ----------------------------------------------------------------------
# Plain-language Q&A
# ----------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=8000)
    # Narrow retrieval to particular documents or departments.
    departments: list[str] = Field(default_factory=list)
    doc_keys: list[str] = Field(default_factory=list)
    include_superseded: bool = False
    # Consult other internal AI systems. None = follow the platform default.
    consult_peers: bool | None = None
    peers: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=30)
    conversation_id: str | None = None
    # Identifies the calling in-house application, for audit and analytics.
    app: str = ""

    @field_validator("question")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class Reference(BaseModel):
    marker: str
    type: Literal["document", "ai_system"]
    citation: str
    document_key: str | None = None
    document_title: str | None = None
    revision: str | None = None
    department: str | None = None
    section: str | None = None
    section_path: str | None = None
    heading: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_uri: str | None = None
    score: float | None = None
    chunk_id: str | None = None
    system: str | None = None
    display_name: str | None = None
    model: str | None = None
    queried_at: str | None = None
    latency_ms: float | None = None
    peer_references: list[dict] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    references: list[Reference] = Field(default_factory=list)
    confidence: float
    grounded: bool
    passages_considered: int = 0
    peers_consulted: list[str] = Field(default_factory=list)
    conversation_id: str | None = None
    model: str = ""
    latency_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    departments: list[str] = Field(default_factory=list)
    doc_keys: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    include_text: bool = True


class SearchHit(BaseModel):
    citation: str
    document_key: str
    document_title: str
    revision: str
    section: str
    section_path: str
    page_start: int | None
    page_end: int | None
    score: float
    text: str = ""


# ----------------------------------------------------------------------
# Documents
# ----------------------------------------------------------------------

class DocumentSummary(BaseModel):
    id: str
    doc_key: str
    title: str
    department: str
    doc_type: str
    revision: str
    effective_date: date | None
    review_due_date: date | None
    status: str
    page_count: int
    chunk_count: int
    allowed_groups: list[str]
    classification: str
    updated_at: datetime | None = None


class IngestMetadata(BaseModel):
    doc_key: str = Field(..., min_length=1, max_length=128)
    title: str = ""
    department: str = ""
    doc_type: str = "governing"
    revision: str = ""
    effective_date: date | None = None
    review_due_date: date | None = None
    allowed_groups: list[str] = Field(default_factory=list)
    classification: str = "internal"
    force: bool = False


class IngestResponse(BaseModel):
    document_id: str
    doc_key: str
    revision: str
    chunk_count: int
    page_count: int
    skipped: bool = False
    reason: str = ""


# ----------------------------------------------------------------------
# Maintenance
# ----------------------------------------------------------------------

class UtilizationEntry(BaseModel):
    tail_number: str
    day: date
    flight_hours: float = Field(default=0.0, ge=0)
    cycles: int = Field(default=0, ge=0)
    landings: int = Field(default=0, ge=0)
    source: str = "api"


class ForecastResponse(BaseModel):
    tail_number: str
    model: str = ""
    serial_number: str = ""
    as_of: date
    utilization: dict[str, Any]
    forecasts: list[dict]
    summary: dict[str, Any]


class PlanRequest(BaseModel):
    tail_numbers: list[str] = Field(default_factory=list)
    horizon_days: int | None = Field(default=None, ge=1, le=1095)
    # Persist the generated plan as schedule events.
    commit: bool = False
    explain: bool = False


class CancelEventRequest(BaseModel):
    reason: str = Field(..., min_length=3, max_length=1000)
    reschedule: bool = True


class DeferRequest(BaseModel):
    tail_number: str
    task_code: str
    until: date
    reason: str = Field(..., min_length=3, max_length=1000)
    extension_flight_hours: float = Field(default=0.0, ge=0)
    authority_reference: str = ""


class CompleteEventRequest(BaseModel):
    completed_on: date | None = None
    work_order: str = ""
    completed_task_codes: list[str] | None = None


class RescheduleRequest(BaseModel):
    new_start: date
    reason: str = ""


# ----------------------------------------------------------------------
# LaTeX
# ----------------------------------------------------------------------

class LatexRequest(BaseModel):
    brief: str = Field(default="", max_length=20000)
    source: str = Field(default="", max_length=2_000_000)
    template: str = ""
    document_class: str = "article"
    # Ground the document in the governing documents and cite them.
    ground_in_documents: bool = False
    departments: list[str] = Field(default_factory=list)
    compile: bool = True

    @field_validator("document_class")
    @classmethod
    def _safe_class(cls, v: str) -> str:
        allowed = {"article", "report", "book", "letter", "memoir", "scrartcl"}
        if v not in allowed:
            raise ValueError(f"document_class must be one of: {', '.join(sorted(allowed))}")
        return v


class LatexResponse(BaseModel):
    source: str
    compiled: bool
    artifact_id: str = ""
    download_url: str = ""
    page_count: int = 0
    size_bytes: int = 0
    errors: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)


# ----------------------------------------------------------------------
# Development assistance
# ----------------------------------------------------------------------

class DevRequest(BaseModel):
    instruction: str = Field(..., min_length=3, max_length=20000)
    # Repo-relative paths to include as context.
    files: list[str] = Field(default_factory=list)
    workspace: str = ""
    language: str = ""
    mode: Literal["explain", "generate", "review", "patch", "test"] = "generate"


class DevResponse(BaseModel):
    output: str
    files_included: list[str] = Field(default_factory=list)
    truncated: list[str] = Field(default_factory=list)
    model: str = ""
    latency_ms: float = 0.0


# ----------------------------------------------------------------------
# Administration
# ----------------------------------------------------------------------

class ApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    description: str = ""
    scopes: list[str] = Field(default_factory=list)
    groups: list[str] = Field(default_factory=list)
    owner_email: str = ""
    expires_at: datetime | None = None
    rate_limit_per_minute: int = Field(default=120, ge=1, le=10000)


class ApiKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    scopes: list[str]
    groups: list[str]
    active: bool
    expires_at: datetime | None = None
    # Present only in the creation response.
    api_key: str | None = None


class PeerRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=128)
    base_url: str
    protocol: Literal["openai", "anthropic", "rest"] = "openai"
    display_name: str = ""
    description: str = ""
    model: str = ""
    auth_type: Literal["none", "bearer", "api_key", "client_credentials"] = "none"
    auth_header: str = "Authorization"
    auth_env_var: str = ""
    token_url: str = ""
    scope: str = ""
    capabilities: list[str] = Field(default_factory=list)
    allowed_groups: list[str] = Field(default_factory=list)
    enabled: bool = True
    citable: bool = True
    timeout_s: float = Field(default=45.0, ge=1, le=300)
    meta: dict = Field(default_factory=dict)


class WhoAmIResponse(BaseModel):
    subject: str
    display_name: str
    email: str
    kind: str
    groups: list[str]
    roles: list[str]
    scopes: list[str]
    issuer: str = ""


# ----------------------------------------------------------------------
# Minimum Equipment List
# ----------------------------------------------------------------------

class MelCheckRequest(BaseModel):
    tail_number: str
    # Identify the item directly when it is known. Otherwise describe the
    # defect and the platform returns candidates for a human to choose from.
    item_number: str = ""
    description: str = Field(default="", max_length=2000)
    discovered_on: date | None = None
    quantity_inoperative: int = Field(default=1, ge=1, le=99)
    # e.g. ["IFR", "night", "over water"] - checked against the item's
    # prohibited operations.
    intended_operation: list[str] = Field(default_factory=list)


class MelCandidate(BaseModel):
    item_number: str
    title: str
    category: str
    ata_chapter: str = ""
    system: str = ""


class MelCheckResponse(BaseModel):
    tail_number: str
    # Present when a single item was identified.
    decision: dict | None = None
    # Present when the description matched several items, or none.
    candidates: list[MelCandidate] = Field(default_factory=list)
    needs_clarification: bool = False
    message: str = ""
    references: list[Reference] = Field(default_factory=list)


class MelDeferralRequest(BaseModel):
    tail_number: str
    item_number: str
    defect_description: str = Field(..., min_length=3, max_length=2000)
    # The licensed person accepting the deferral. Defaults to the caller.
    accepted_by: str = ""
    discovered_on: date | None = None
    quantity_inoperative: int = Field(default=1, ge=1, le=99)
    work_order: str = ""
    # Confirmation that the MEL's conditions have actually been carried out.
    placard_fitted: bool = False
    operational_procedure_applied: bool = False
    maintenance_procedure_applied: bool = False
    notes: str = ""


class MelDeferralResponse(BaseModel):
    id: str
    tail_number: str
    item_number: str
    category: str
    defect_description: str
    discovered_on: date
    expires_on: date
    days_remaining: int
    status: str
    accepted_by: str
    extended: bool = False
    citation: str = ""
    conditions: list[str] = Field(default_factory=list)


class MelClearRequest(BaseModel):
    cleared_on: date | None = None
    rectification_notes: str = ""


class MelExtendRequest(BaseModel):
    reason: str = Field(..., min_length=3, max_length=1000)
    authority_reference: str = ""


# ----------------------------------------------------------------------
# Operational reports
# ----------------------------------------------------------------------

class ReportDraftRequest(BaseModel):
    request_text: str = Field(..., min_length=5, max_length=4000)


class ReportDraftResponse(BaseModel):
    request_text: str
    query: str
    explanation: str = ""
    assumptions: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    valid: bool = False
    rejection: str = ""
    repaired: bool = False
    warnings: list[str] = Field(default_factory=list)


class ReportAskRequest(BaseModel):
    """One-shot ad-hoc report: draft, run and return, without saving."""

    request_text: str = Field(..., min_length=5, max_length=4000)
    output_formats: list[str] = Field(default_factory=lambda: ["markdown"])
    narrative: bool = True


class ReportCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=200)
    request_text: str = Field(..., min_length=5, max_length=4000)
    query: str = Field(..., min_length=5)
    description: str = ""
    source: str = "namis"
    query_language: Literal["sql", "rest"] = "sql"
    parameters: dict = Field(default_factory=dict)
    output_formats: list[str] = Field(default_factory=lambda: ["markdown", "csv"])
    narrative: bool = True
    # AD groups permitted to run this report and read its results.
    allowed_groups: list[str] = Field(default_factory=list)


class ReportUpdateRequest(BaseModel):
    description: str | None = None
    request_text: str | None = None
    query: str | None = None
    parameters: dict | None = None
    output_formats: list[str] | None = None
    narrative: bool | None = None
    allowed_groups: list[str] | None = None


class ReportScheduleRequest(BaseModel):
    # Five-field cron, or an @alias such as @daily.
    cron: str = Field(..., min_length=1, max_length=128)
    timezone: str = "UTC"
    enabled: bool = True


class ReportSummary(BaseModel):
    id: str
    name: str
    description: str = ""
    request_text: str
    status: str
    query_language: str = "sql"
    output_formats: list[str] = Field(default_factory=list)
    narrative: bool = True
    allowed_groups: list[str] = Field(default_factory=list)
    owner: str = ""
    approved_by: str = ""
    approved_at: datetime | None = None
    approval_current: bool = False
    schedule_cron: str = ""
    schedule_timezone: str = "UTC"
    schedule_enabled: bool = False
    schedule_description: str = ""
    last_run_at: datetime | None = None
    next_run_at: datetime | None = None


class ReportRunSummary(BaseModel):
    id: str
    definition_id: str
    status: str
    trigger: str
    actor: str = ""
    started_at: datetime
    finished_at: datetime | None = None
    row_count: int = 0
    truncated: bool = False
    duration_ms: float = 0.0
    narrative: str = ""
    artifacts: list[dict] = Field(default_factory=list)
    error: str = ""
    warnings: list = Field(default_factory=list)
