from datetime import datetime, timezone
import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

# Default domain to ID prefix mapping
DOMAIN_PREFIX_MAP = {
    "artificial_intelligence": "AI",
    "cybersecurity": "CY",
    "agriculture": "AG",
    "healthcare": "HC",
    "climate": "CL",
}


class PaperMetadata(BaseModel):
    """Pydantic model for accepted scientific paper metadata in Research Mind."""

    # Required fields
    paper_id: str = Field(..., description="Stable paper ID (e.g. AI001, HC001)")
    domain: str = Field(..., description="Canonical domain name (e.g. artificial_intelligence)")
    subtopic: str = Field(..., description="Canonical subtopic identifier")
    title: str = Field(..., description="Full paper title")
    authors: List[str] = Field(..., min_length=1, description="List of paper authors")
    publication_year: int = Field(..., ge=1800, description="Year of publication")
    source: str = Field(..., description="Source platform name (e.g. arXiv, PubMed Central, Semantic Scholar)")
    local_path: str = Field(..., description="Relative local path to saved PDF")
    file_hash: str = Field(..., description="SHA-256 hash of PDF file")
    collection_date: str = Field(..., description="ISO 8601 UTC timestamp of collection")
    status: str = Field(default="downloaded", description="Paper status (e.g. downloaded)")

    # Nullable / Optional fields (source-dependent)
    abstract: Optional[str] = Field(default=None, description="Paper abstract text if available")
    publication_date: Optional[str] = Field(default=None, description="Exact publication date YYYY-MM-DD if available")
    venue: Optional[str] = Field(default=None, description="Journal or conference venue if available")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier if available")
    arxiv_id: Optional[str] = Field(default=None, description="arXiv identifier if available")
    source_id: Optional[str] = Field(default=None, description="Source platform native paper ID if available")
    source_url: Optional[str] = Field(default=None, description="URL to source landing page if available")
    pdf_url: Optional[str] = Field(default=None, description="URL to open access PDF if available")
    license: Optional[str] = Field(default=None, description="Open access license string if available")
    open_access: Optional[bool] = Field(default=True, description="Open access indicator")

    @field_validator("publication_year")
    @classmethod
    def validate_publication_year(cls, v: int) -> int:
        max_year = datetime.now(timezone.utc).year + 1
        if v > max_year:
            raise ValueError(f"Publication year {v} exceeds maximum allowed year ({max_year})")
        return v

    @field_validator("title", "domain", "subtopic", "source", "local_path", "file_hash")
    @classmethod
    def validate_non_empty_str(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty string")
        return v.strip()

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, v: List[str]) -> List[str]:
        cleaned = [a.strip() for a in v if a and a.strip()]
        if not cleaned:
            raise ValueError("Authors list must contain at least one non-empty author name")
        return cleaned

    @model_validator(mode="after")
    def validate_paper_id_prefix(self) -> "PaperMetadata":
        """Validate paper_id prefix matches the domain prefix rule."""
        domain_name = self.domain.strip()
        pid = self.paper_id.strip()

        # Check known domain prefix mapping
        if domain_name in DOMAIN_PREFIX_MAP:
            expected_prefix = DOMAIN_PREFIX_MAP[domain_name]
            # ID must start with expected prefix followed by digits (e.g. AI001)
            pattern = rf"^{expected_prefix}\d+$"
            if not re.match(pattern, pid):
                raise ValueError(
                    f"Invalid paper_id '{pid}' for domain '{domain_name}'. "
                    f"Must match prefix '{expected_prefix}' followed by numbers (e.g. '{expected_prefix}001')."
                )

        return self
