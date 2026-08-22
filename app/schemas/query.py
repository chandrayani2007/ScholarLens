"""
Pydantic Schemas for Research Query Requests
"""

from typing import Optional
from pydantic import BaseModel, Field


class ResearchQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language research question")
    domain: Optional[str] = Field(None, description="Optional domain filter (e.g. artificial_intelligence, cybersecurity, agriculture, healthcare, climate)")
    subtopic: Optional[str] = Field(None, description="Optional subtopic filter")
    paper_id: Optional[str] = Field(None, description="Optional paper ID filter (e.g. AI001)")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Top-K evidence passages to retrieve")
