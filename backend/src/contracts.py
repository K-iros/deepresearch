"""Structured contracts shared across planner, summarizer, reporter and streaming."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

CONTRACT_VERSION = "1.0.0"


class RetryPolicy(BaseModel):
    """Retry semantics for one task."""

    max_attempts: int = Field(default=2, ge=1, le=5)
    backoff_seconds: float = Field(default=0.0, ge=0.0, le=30.0)


class PlannerTaskContract(BaseModel):
    """Single planner task contract."""

    title: str = Field(..., min_length=1, max_length=120)
    intent: str = Field(..., min_length=1, max_length=500)
    query: str = Field(..., min_length=1, max_length=300)
    depends_on: list[int] = Field(default_factory=list)
    priority: int = Field(default=50, ge=0, le=100)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)


class PlannerOutputContract(BaseModel):
    """Versioned planner output envelope."""

    version: str = Field(default=CONTRACT_VERSION)
    tasks: list[PlannerTaskContract] = Field(default_factory=list)


class TaskSummaryContract(BaseModel):
    """Versioned task summary contract."""

    version: str = Field(default=CONTRACT_VERSION)
    task_id: int = Field(..., ge=1)
    summary_markdown: str = Field(..., min_length=1)
    key_findings: list[str] = Field(default_factory=list)


class ReportContract(BaseModel):
    """Versioned final report contract."""

    version: str = Field(default=CONTRACT_VERSION)
    report_markdown: str = Field(..., min_length=1)
    sections: dict[str, str] = Field(default_factory=dict)


class SearchDocumentContract(BaseModel):
    """Unified search result document for downstream summarization."""

    title: str = Field(default="")
    summary: str = Field(default="")
    url: Optional[str] = Field(default=None)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_type: Literal["web", "news", "academic", "other"] = Field(default="web")
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_content: str = Field(default="")

    @field_validator("summary", "raw_content", mode="before")
    @classmethod
    def _coerce_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value)


class SearchBundleContract(BaseModel):
    """Normalized search payload independent from upstream provider shape."""

    version: str = Field(default=CONTRACT_VERSION)
    backend: str = Field(default="unknown")
    answer: Optional[str] = Field(default=None)
    notices: list[str] = Field(default_factory=list)
    documents: list[SearchDocumentContract] = Field(default_factory=list)


class EventEnvelopeContract(BaseModel):
    """Canonical stream event model."""

    event_type: str = Field(..., min_length=1)
    run_id: str = Field(..., min_length=1)
    trace_id: str = Field(..., min_length=1)
    sequence: int = Field(..., ge=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_id: Optional[int] = Field(default=None, ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
