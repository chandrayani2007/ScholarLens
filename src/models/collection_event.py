from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class EventAction(str, Enum):
    """Supported event actions for dataset collection auditing."""
    CANDIDATE_DISCOVERED = "candidate_discovered"
    DUPLICATE_DETECTED = "duplicate_detected"
    DOWNLOAD_STARTED = "download_started"
    DOWNLOAD_SUCCESS = "download_success"
    DOWNLOAD_FAILED = "download_failed"
    VALIDATION_FAILED = "validation_failed"
    REJECTED = "rejected"
    ACCEPTED = "accepted"


class CollectionEvent(BaseModel):
    """Pydantic model recording an audit event during paper collection."""

    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of event")
    domain: str = Field(..., description="Canonical domain name")
    subtopic: str = Field(..., description="Canonical subtopic name")
    source: str = Field(..., description="Source platform name (e.g. arXiv, PubMed Central)")
    action: EventAction = Field(..., description="Event action type")

    # Optional / Contextual fields (null when unavailable)
    source_id: Optional[str] = Field(default=None, description="Native source ID if available")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier if available")
    title: Optional[str] = Field(default=None, description="Paper title if available")
    reason: Optional[str] = Field(default=None, description="Reason for rejection, duplicate, or failure")
    paper_id: Optional[str] = Field(default=None, description="Assigned paper ID if accepted or processing")

    @field_validator("timestamp", "domain", "subtopic", "source")
    @classmethod
    def validate_non_empty_str(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty string")
        return v.strip()
