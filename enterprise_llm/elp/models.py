"""
ORM models for the Enterprise LLM Platform.

Four groups of tables:

* **Knowledge**  - ``documents`` / ``doc_chunks`` hold the department
  governing documents and their embeddings, with group-based ACLs.
* **Identity**   - ``api_keys`` (service accounts for in-house apps) and
  ``audit_events`` (who asked what, which sources answered).
* **Federation** - ``peer_systems`` registers other internal LLM/AI services.
* **Maintenance**- the aircraft, task-card, compliance and event tables that
  drive predictive scheduling.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .config import get_settings
from .db import Base

EMBED_DIM = get_settings().inference.embed_dim


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


# ======================================================================
# Knowledge base
# ======================================================================

class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class Document(Base, TimestampMixin):
    """A department governing document (manual, policy, MPD, SOP...)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Stable human-facing handle used in citations, e.g. "OPS-MAN-001".
    doc_key: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    department: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    doc_type: Mapped[str] = mapped_column(String(64), default="governing", nullable=False)
    # Revision/issue as printed on the document itself - essential for
    # answering "which revision says that?".
    revision: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    effective_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    review_due_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    source_uri: Mapped[str] = mapped_column(Text, default="", nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Access control: AD groups permitted to retrieve from this document.
    # Empty array => readable by any authenticated caller.
    allowed_groups: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    classification: Mapped[str] = mapped_column(
        String(32), default="internal", nullable=False
    )

    status: Mapped[str] = mapped_column(
        String(24), default=DocumentStatus.PENDING.value, nullable=False
    )
    status_detail: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Newer revision that replaces this one.
    superseded_by: Mapped[str | None] = mapped_column(String(36), nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    chunks: Mapped[list[DocChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        UniqueConstraint("doc_key", "revision", name="uq_documents_key_revision"),
        Index("ix_documents_status", "status"),
        Index("ix_documents_department", "department"),
    )


class DocChunk(Base):
    """A retrievable passage carrying enough structure to cite precisely."""

    __tablename__ = "doc_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    # Structural anchors -> "OPS-MAN-001 Rev C, section 4.2.3, p. 51".
    section_number: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    section_path: Mapped[str] = mapped_column(Text, default="", nullable=False)
    heading: Mapped[str] = mapped_column(Text, default="", nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks", lazy="joined")

    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunk_doc_ordinal"),
        Index("ix_doc_chunks_document", "document_id"),
    )

    def citation_label(self) -> str:
        doc = self.document
        parts = [doc.doc_key]
        if doc.revision:
            parts.append(f"Rev {doc.revision}")
        if self.section_number:
            parts.append(f"§{self.section_number}")
        elif self.heading:
            parts.append(self.heading[:60])
        if self.page_start:
            page = (
                f"p. {self.page_start}"
                if not self.page_end or self.page_end == self.page_start
                else f"pp. {self.page_start}-{self.page_end}"
            )
            parts.append(page)
        return ", ".join(parts)


# ======================================================================
# Identity, access and audit
# ======================================================================

class ApiKey(Base, TimestampMixin):
    """Service account credential for an in-house application."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Only a hash is stored; the plaintext key is shown once at creation.
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    key_prefix: Mapped[str] = mapped_column(String(24), nullable=False)

    # Scopes granted to the application (see auth.scopes).
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    # AD groups the service account inherits, so document ACLs still apply.
    groups: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list, nullable=False)
    owner_email: Mapped[str] = mapped_column(String(255), default="", nullable=False)

    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Simple per-minute throttle enforced by the gateway.
    rate_limit_per_minute: Mapped[int] = mapped_column(Integer, default=120, nullable=False)


class AuditEvent(Base):
    """Immutable record of every privileged action and answered question."""

    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    request_id: Mapped[str] = mapped_column(String(36), default="", nullable=False)
    actor: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    actor_type: Mapped[str] = mapped_column(String(24), default="user", nullable=False)
    actor_groups: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource: Mapped[str] = mapped_column(Text, default="", nullable=False)
    outcome: Mapped[str] = mapped_column(String(24), default="ok", nullable=False)
    client_ip: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    detail: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    __table_args__ = (
        Index("ix_audit_ts", "ts"),
        Index("ix_audit_actor", "actor"),
        Index("ix_audit_action", "action"),
    )


class Conversation(Base, TimestampMixin):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(Text, default="", nullable=False)
    app: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_conversations_actor", "actor"),)


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Structured citations returned with an assistant turn.
    references: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (Index("ix_messages_conversation", "conversation_id"),)


# ======================================================================
# Federation
# ======================================================================

class PeerSystem(Base, TimestampMixin):
    """Another internal LLM/AI service this platform can consult."""

    __tablename__ = "peer_systems"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # openai | rest | mcp
    protocol: Mapped[str] = mapped_column(String(24), default="openai", nullable=False)
    base_url: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(128), default="", nullable=False)

    # none | bearer | api_key | client_credentials
    auth_type: Mapped[str] = mapped_column(String(32), default="none", nullable=False)
    auth_header: Mapped[str] = mapped_column(String(64), default="Authorization", nullable=False)
    # Name of the environment variable holding the secret - never the secret.
    auth_env_var: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    token_url: Mapped[str] = mapped_column(Text, default="", nullable=False)
    scope: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Topic tags used to route questions to the right peer.
    capabilities: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    # AD groups allowed to trigger a call to this peer.
    allowed_groups: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    citable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    timeout_s: Mapped[float] = mapped_column(Float, default=45.0, nullable=False)
    # Rolling health signal maintained by the orchestrator.
    last_ok_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str] = mapped_column(Text, default="", nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)


# ======================================================================
# Aviation maintenance
# ======================================================================

class IntervalBasis(str, enum.Enum):
    FLIGHT_HOURS = "flight_hours"
    CYCLES = "cycles"
    LANDINGS = "landings"
    CALENDAR_DAYS = "calendar_days"


class EventStatus(str, enum.Enum):
    FORECAST = "forecast"
    PLANNED = "planned"
    CONFIRMED = "confirmed"
    IN_WORK = "in_work"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Aircraft(Base, TimestampMixin):
    __tablename__ = "aircraft"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tail_number: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    model: Mapped[str] = mapped_column(String(64), default="AW139", nullable=False)
    serial_number: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    # SN / LN / ENH / PLUS, matching the existing configuration resolver.
    configuration: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    base_station: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    operator: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    in_service: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Current counters and the moment they were read.
    current_flight_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    current_cycles: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_landings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    counters_as_of: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    utilization: Mapped[list[UtilizationRecord]] = relationship(
        back_populates="aircraft", cascade="all, delete-orphan"
    )


class UtilizationRecord(Base):
    """One day of recorded usage, used to forecast future consumption."""

    __tablename__ = "utilization_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    aircraft_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("aircraft.id", ondelete="CASCADE"), nullable=False
    )
    day: Mapped[date] = mapped_column(Date, nullable=False)
    flight_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    cycles: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    landings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source: Mapped[str] = mapped_column(String(64), default="manual", nullable=False)

    aircraft: Mapped[Aircraft] = relationship(back_populates="utilization")

    __table_args__ = (
        UniqueConstraint("aircraft_id", "day", name="uq_utilization_aircraft_day"),
        Index("ix_utilization_day", "day"),
    )


class MaintenanceTask(Base, TimestampMixin):
    """
    A task card from the customer-supplied standard maintenance schedule.

    Intervals are multi-dimensional: a task may be due every 600 flight
    hours *or* 12 months, whichever comes first.  Any subset may be set.
    """

    __tablename__ = "maintenance_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    task_code: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    ata_chapter: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    # inspection | replacement | lubrication | ad | sb | overhaul | check
    task_type: Mapped[str] = mapped_column(String(32), default="inspection", nullable=False)

    # Applicability - empty list means "all".
    applicable_models: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    applicable_configurations: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    applicable_serials: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )

    # Repeat intervals.
    interval_flight_hours: Mapped[float | None] = mapped_column(Float)
    interval_cycles: Mapped[int | None] = mapped_column(Integer)
    interval_landings: Mapped[int | None] = mapped_column(Integer)
    interval_calendar_days: Mapped[int | None] = mapped_column(Integer)

    # Permitted overrun beyond the nominal due point.
    tolerance_flight_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    tolerance_calendar_days: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Planning inputs.
    estimated_man_hours: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    estimated_downtime_hours: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    technicians_required: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    required_skills: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    required_parts: Mapped[list] = mapped_column(JSONB, default=list, nullable=False)
    requires_hangar: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Regulatory character.
    is_airworthiness_limitation: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    can_be_deferred: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    max_deferral_days: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Traceability back into the governing documents.
    source_document_key: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    source_reference: Mapped[str] = mapped_column(Text, default="", nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    __table_args__ = (Index("ix_tasks_ata", "ata_chapter"),)

    def applies_to(self, aircraft: Aircraft) -> bool:
        if not self.active:
            return False
        if self.applicable_models and aircraft.model not in self.applicable_models:
            return False
        if (
            self.applicable_configurations
            and aircraft.configuration not in self.applicable_configurations
        ):
            return False
        return not (
            self.applicable_serials
            and aircraft.serial_number not in self.applicable_serials
        )


class ComplianceRecord(Base):
    """When a task was last accomplished on a given aircraft."""

    __tablename__ = "compliance_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    aircraft_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("aircraft.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("maintenance_tasks.id", ondelete="CASCADE"), nullable=False
    )
    completed_on: Mapped[date] = mapped_column(Date, nullable=False)
    at_flight_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    at_cycles: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    at_landings: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    work_order: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    performed_by: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    aircraft: Mapped[Aircraft] = relationship()
    task: Mapped[MaintenanceTask] = relationship()

    __table_args__ = (
        Index("ix_compliance_aircraft_task", "aircraft_id", "task_id"),
        Index("ix_compliance_completed", "completed_on"),
    )


class MaintenanceEvent(Base, TimestampMixin):
    """A scheduled visit bundling one or more due task cards."""

    __tablename__ = "maintenance_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    aircraft_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("aircraft.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, default="", nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), default=EventStatus.PLANNED.value, nullable=False
    )
    scheduled_start: Mapped[date] = mapped_column(Date, nullable=False)
    scheduled_end: Mapped[date] = mapped_column(Date, nullable=False)
    station: Mapped[str] = mapped_column(String(64), default="", nullable=False)

    estimated_man_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    estimated_downtime_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    # Populated when status becomes CANCELLED.
    cancellation_reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancelled_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    # Event created to absorb the tasks from a cancelled event.
    replaces_event_id: Mapped[str | None] = mapped_column(String(36))

    created_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    aircraft: Mapped[Aircraft] = relationship()
    items: Mapped[list[EventTask]] = relationship(
        back_populates="event", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        Index("ix_events_aircraft_start", "aircraft_id", "scheduled_start"),
        Index("ix_events_status", "status"),
    )


class EventTask(Base):
    """A task card assigned to a maintenance event."""

    __tablename__ = "event_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    event_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("maintenance_events.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("maintenance_tasks.id", ondelete="CASCADE"), nullable=False
    )
    # Snapshot of the forecast at planning time, for later audit.
    due_on: Mapped[date | None] = mapped_column(Date)
    hard_limit_on: Mapped[date | None] = mapped_column(Date)
    driving_basis: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    remaining_margin: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # planned | completed | removed | deferred
    status: Mapped[str] = mapped_column(String(24), default="planned", nullable=False)

    event: Mapped[MaintenanceEvent] = relationship(back_populates="items")
    task: Mapped[MaintenanceTask] = relationship(lazy="joined")

    __table_args__ = (
        UniqueConstraint("event_id", "task_id", name="uq_event_task"),
    )


class Deferral(Base, TimestampMixin):
    """An approved extension of a task beyond its nominal due point."""

    __tablename__ = "deferrals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    aircraft_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("aircraft.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("maintenance_tasks.id", ondelete="CASCADE"), nullable=False
    )
    event_id: Mapped[str | None] = mapped_column(String(36))
    reason: Mapped[str] = mapped_column(Text, default="", nullable=False)
    approved_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    approved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    expires_on: Mapped[date] = mapped_column(Date, nullable=False)
    extension_flight_hours: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    released: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    authority_reference: Mapped[str] = mapped_column(Text, default="", nullable=False)

    __table_args__ = (
        Index("ix_deferrals_aircraft_task", "aircraft_id", "task_id"),
    )


# ======================================================================
# Minimum Equipment List / Configuration Deviation List
# ======================================================================

class MelCategory(str, enum.Enum):
    """
    Rectification interval categories.

    B, C and D carry fixed intervals; A means "as stated in the remarks
    column of the MEL itself", which varies item by item.
    """

    A = "A"
    B = "B"      # 3 consecutive calendar days
    C = "C"      # 10 consecutive calendar days
    D = "D"      # 120 consecutive calendar days
    # Configuration Deviation List items: missing secondary airframe parts.
    CDL = "CDL"


class DeferralStatus(str, enum.Enum):
    OPEN = "open"
    CLEARED = "cleared"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class MelItem(Base, TimestampMixin):
    """
    One entry from the operator's approved Minimum Equipment List.

    The MEL is what makes an aircraft dispatchable with something
    inoperative.  Anything not listed here must work - that is the whole
    premise of the document, and the dispatch engine enforces it.
    """

    __tablename__ = "mel_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # ATA-style reference as printed in the MEL, e.g. "24-11-01".
    item_number: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    ata_chapter: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    system: Mapped[str] = mapped_column(Text, default="", nullable=False)

    category: Mapped[str] = mapped_column(String(8), nullable=False)
    # Category A items carry their interval in the remarks; this is the
    # number of days parsed from (or entered against) that text.
    category_a_days: Mapped[int | None] = mapped_column(Integer)

    # "2 installed, 1 required for dispatch" => one may be deferred.
    number_installed: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    number_required: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Conditions from the MEL's remarks column.
    remarks: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # (o) operational and (m) maintenance procedures that must be applied.
    operational_procedure: Mapped[str] = mapped_column(Text, default="", nullable=False)
    maintenance_procedure: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # Performance or operational penalty, common on CDL items.
    placard_text: Mapped[str] = mapped_column(Text, default="", nullable=False)
    performance_penalty: Mapped[str] = mapped_column(Text, default="", nullable=False)

    # Item numbers that may not be inoperative at the same time as this one.
    incompatible_with: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    # Operations this item forbids, e.g. "IFR", "night", "over water".
    prohibited_operations: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )

    # Applicability, mirroring MaintenanceTask.
    applicable_models: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    applicable_configurations: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )
    applicable_serials: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list, nullable=False
    )

    # Whether the authority permits a one-time extension of the interval.
    extension_permitted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # Traceability to the approved document and its revision.
    source_document_key: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    source_revision: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    source_reference: Mapped[str] = mapped_column(Text, default="", nullable=False)

    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    __table_args__ = (
        UniqueConstraint("item_number", "source_revision", name="uq_mel_item_revision"),
        Index("ix_mel_items_number", "item_number"),
        Index("ix_mel_items_ata", "ata_chapter"),
    )

    def applies_to(self, aircraft: Aircraft) -> bool:
        if not self.active:
            return False
        if self.applicable_models and aircraft.model not in self.applicable_models:
            return False
        if (
            self.applicable_configurations
            and aircraft.configuration not in self.applicable_configurations
        ):
            return False
        return not (
            self.applicable_serials
            and aircraft.serial_number not in self.applicable_serials
        )


class MelDeferral(Base, TimestampMixin):
    """
    An open deferred defect: something inoperative that the MEL permits.

    Distinct from ``Deferral``, which extends a *scheduled maintenance task*
    past its due point.  This one records unserviceable equipment carried
    under MEL relief, and its expiry is an airworthiness limit: past it the
    aircraft may not be dispatched.
    """

    __tablename__ = "mel_deferrals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    aircraft_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("aircraft.id", ondelete="CASCADE"), nullable=False
    )
    mel_item_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("mel_items.id", ondelete="RESTRICT"), nullable=False
    )
    # Denormalised so the record stays readable if the MEL is revised.
    item_number: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(8), nullable=False)

    defect_description: Mapped[str] = mapped_column(Text, nullable=False)
    discovered_on: Mapped[date] = mapped_column(Date, nullable=False)
    # Last day the aircraft may be dispatched with this item inoperative.
    expires_on: Mapped[date] = mapped_column(Date, nullable=False)
    original_expires_on: Mapped[date] = mapped_column(Date, nullable=False)

    quantity_inoperative: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=DeferralStatus.OPEN.value, nullable=False
    )

    raised_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    # Licensed engineer who accepted the deferral.
    accepted_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    work_order: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    placard_fitted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    operational_procedure_applied: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    maintenance_procedure_applied: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )

    # One-time extension, where the authority permits it.
    extended: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    extension_approved_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    extension_authority_reference: Mapped[str] = mapped_column(
        Text, default="", nullable=False
    )
    extended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    cleared_on: Mapped[date | None] = mapped_column(Date)
    cleared_by: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    rectification_notes: Mapped[str] = mapped_column(Text, default="", nullable=False)

    notes: Mapped[str] = mapped_column(Text, default="", nullable=False)
    meta: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)

    aircraft: Mapped[Aircraft] = relationship()
    mel_item: Mapped[MelItem] = relationship(lazy="joined")

    __table_args__ = (
        Index("ix_mel_deferrals_aircraft_status", "aircraft_id", "status"),
        Index("ix_mel_deferrals_expires", "expires_on"),
    )
