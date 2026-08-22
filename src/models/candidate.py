from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class PaperCandidate(BaseModel):
    """Pydantic model representing a candidate paper discovered from a scholarly API."""

    title: str = Field(..., description="Raw paper title from API")
    authors: List[str] = Field(default_factory=list, description="List of author names")
    domain: str = Field(..., description="Canonical domain name")
    subtopic: str = Field(..., description="Canonical subtopic name")
    search_query: str = Field(..., description="Search query string that discovered this candidate")
    source: str = Field(..., description="Source platform name (e.g. arXiv, Europe PMC, Semantic Scholar)")

    # Nullable / Source-dependent metadata
    abstract: Optional[str] = Field(default=None, description="Paper abstract text if available")
    publication_year: Optional[int] = Field(default=None, description="Year of publication")
    publication_date: Optional[str] = Field(default=None, description="Exact publication date string YYYY-MM-DD")
    venue: Optional[str] = Field(default=None, description="Publication venue or journal name")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier if available")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv ID if available")
    source_id: Optional[str] = Field(default=None, description="Native source ID (e.g. PMID, PMCID, S2 ID)")
    source_url: Optional[str] = Field(default=None, description="URL to paper landing page")
    pdf_url: Optional[str] = Field(default=None, description="Direct URL to open access PDF if available")
    license: Optional[str] = Field(default=None, description="License string if available")
    open_access: Optional[bool] = Field(default=None, description="Open access indicator if declared")

    @field_validator("title", "domain", "subtopic", "search_query", "source")
    @classmethod
    def validate_non_empty_str(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty string")
        return v.strip()

    @field_validator("authors")
    @classmethod
    def clean_authors(cls, v: List[str]) -> List[str]:
        return [a.strip() for a in v if a and a.strip()]
