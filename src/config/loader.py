import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

REQUIRED_CANONICAL_DOMAINS = {
    "artificial_intelligence",
    "cybersecurity",
    "agriculture",
    "healthcare",
    "climate",
}


class SubtopicConfig(BaseModel):
    """Configuration model for a single domain subtopic."""
    name: str = Field(..., description="Human readable display name")
    canonical: str = Field(..., description="Unique snake_case identifier")
    target_count: int = Field(default=5, ge=1, description="Target paper count for balancing")
    search_queries: List[str] = Field(..., min_length=1, description="Search queries for paper discovery")

    @field_validator("name", "canonical")
    @classmethod
    def validate_non_empty_str(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Subtopic string fields cannot be empty")
        return v.strip()


class DomainConfig(BaseModel):
    """Configuration model for a canonical domain."""
    canonical_name: str = Field(..., description="Canonical domain identifier")
    display_name: str = Field(..., description="Human readable domain display name")
    id_prefix: str = Field(..., description="Stable paper ID prefix (e.g. AI, CY)")
    target_count: int = Field(default=50, ge=1, description="Target paper count for the domain")
    subtopics: List[SubtopicConfig] = Field(..., min_length=1, description="List of subtopic configurations")

    @field_validator("id_prefix")
    @classmethod
    def validate_id_prefix(cls, v: str) -> str:
        v_clean = v.strip().upper()
        if not v_clean or not v_clean.isalpha():
            raise ValueError(f"ID prefix must be non-empty alphabetic string, got '{v}'")
        return v_clean


class AppConfig(BaseModel):
    """Root configuration model holding all domains."""
    domains: Dict[str, DomainConfig] = Field(..., description="Map of canonical domain name to domain config")

    @model_validator(mode="after")
    def validate_domains_and_prefixes(self) -> "AppConfig":
        # Check all 5 canonical domains exist
        configured_domains = set(self.domains.keys())
        missing_domains = REQUIRED_CANONICAL_DOMAINS - configured_domains
        if missing_domains:
            raise ValueError(f"Missing required canonical domains: {sorted(list(missing_domains))}")

        prefixes = {}
        for key, domain in self.domains.items():
            # Validate key matches domain's canonical_name
            if key != domain.canonical_name:
                raise ValueError(f"Domain key '{key}' does not match canonical_name '{domain.canonical_name}'")

            # Check unique prefix
            prefix = domain.id_prefix
            if prefix in prefixes:
                raise ValueError(
                    f"Duplicate ID prefix '{prefix}' found in domain '{key}' (already used by '{prefixes[prefix]}')"
                )
            prefixes[prefix] = key

        return self


class ConfigLoader:
    """Helper class to load and validate domain configuration from JSON."""

    @staticmethod
    def load_from_file(config_path: Path) -> AppConfig:
        """Load, parse, and validate domain configuration from a JSON file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return AppConfig.model_validate(data)


def load_domain_config(config_path: Optional[Path] = None) -> AppConfig:
    """Utility function to load domain config using default or specified path."""
    if config_path is None:
        # Default path relative to project root
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "config" / "domains.json"

    return ConfigLoader.load_from_file(config_path)
