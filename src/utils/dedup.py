from enum import Enum
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field
from src.utils.text import levenshtein_similarity, normalize_title


class DeduplicationSignalType(str, Enum):
    EXACT_DOI = "exact_doi"
    EXACT_SOURCE_ID = "exact_source_id"
    EXACT_TITLE = "exact_title"
    FILE_HASH = "file_hash"
    FUZZY_TITLE_MATCH = "fuzzy_title_match"
    NONE = "none"


class DeduplicationSignal(BaseModel):
    """Result model for a candidate paper deduplication check."""
    is_duplicate: bool = Field(default=False, description="True if confirmed strong duplicate")
    requires_flag: bool = Field(default=False, description="True if probable duplicate requiring manual/secondary comparison")
    signal_type: DeduplicationSignalType = Field(default=DeduplicationSignalType.NONE)
    matching_paper_id: Optional[str] = Field(default=None, description="Existing paper_id that matched")
    similarity_score: Optional[float] = Field(default=None, description="Similarity score for fuzzy match")
    reason: str = Field(default="No duplicate detected")


def evaluate_candidate_deduplication(
    candidate_title: str,
    candidate_doi: Optional[str] = None,
    candidate_source_id: Optional[str] = None,
    candidate_file_hash: Optional[str] = None,
    existing_dois: Optional[Dict[str, str]] = None,
    existing_source_ids: Optional[Dict[str, str]] = None,
    existing_normalized_titles: Optional[Dict[str, str]] = None,
    existing_file_hashes: Optional[Dict[str, str]] = None,
    fuzzy_threshold: float = 0.88,
) -> DeduplicationSignal:
    """
    Evaluate candidate paper against existing paper corpus using multi-signal hierarchy:
    1. Exact DOI -> Strong duplicate
    2. Exact Source ID -> Strong duplicate
    3. Exact Normalized Title -> Strong duplicate
    4. Exact File Hash -> Strong duplicate
    5. Fuzzy Title Similarity >= fuzzy_threshold -> Probable duplicate (flagged, NOT auto-rejected)
    """
    existing_dois = existing_dois or {}
    existing_source_ids = existing_source_ids or {}
    existing_normalized_titles = existing_normalized_titles or {}
    existing_file_hashes = existing_file_hashes or {}

    # Signal 1: Exact DOI
    if candidate_doi and candidate_doi.strip():
        norm_doi = candidate_doi.strip().lower()
        if norm_doi in existing_dois:
            matching_id = existing_dois[norm_doi]
            return DeduplicationSignal(
                is_duplicate=True,
                requires_flag=False,
                signal_type=DeduplicationSignalType.EXACT_DOI,
                matching_paper_id=matching_id,
                reason=f"Exact DOI match with {matching_id}",
            )

    # Signal 2: Exact Source ID
    if candidate_source_id and candidate_source_id.strip():
        norm_sid = candidate_source_id.strip()
        if norm_sid in existing_source_ids:
            matching_id = existing_source_ids[norm_sid]
            return DeduplicationSignal(
                is_duplicate=True,
                requires_flag=False,
                signal_type=DeduplicationSignalType.EXACT_SOURCE_ID,
                matching_paper_id=matching_id,
                reason=f"Exact source_id match with {matching_id}",
            )

    # Signal 3: Exact Normalized Title
    norm_candidate_title = normalize_title(candidate_title)
    if norm_candidate_title and norm_candidate_title in existing_normalized_titles:
        matching_id = existing_normalized_titles[norm_candidate_title]
        return DeduplicationSignal(
            is_duplicate=True,
            requires_flag=False,
            signal_type=DeduplicationSignalType.EXACT_TITLE,
            matching_paper_id=matching_id,
            reason=f"Exact normalized title match with {matching_id}",
        )

    # Signal 4: File Hash
    if candidate_file_hash and candidate_file_hash.strip():
        norm_hash = candidate_file_hash.strip().lower()
        if norm_hash in existing_file_hashes:
            matching_id = existing_file_hashes[norm_hash]
            return DeduplicationSignal(
                is_duplicate=True,
                requires_flag=False,
                signal_type=DeduplicationSignalType.FILE_HASH,
                matching_paper_id=matching_id,
                reason=f"Exact PDF file hash match with {matching_id}",
            )

    # Signal 5: Fuzzy Title Similarity (Flags candidate, does NOT auto-delete)
    if norm_candidate_title:
        for existing_title, matching_id in existing_normalized_titles.items():
            score = levenshtein_similarity(norm_candidate_title, existing_title)
            if score >= fuzzy_threshold:
                return DeduplicationSignal(
                    is_duplicate=False,
                    requires_flag=True,
                    signal_type=DeduplicationSignalType.FUZZY_TITLE_MATCH,
                    matching_paper_id=matching_id,
                    similarity_score=round(score, 4),
                    reason=f"High title similarity ({round(score * 100, 1)}%) with {matching_id}",
                )

    return DeduplicationSignal(is_duplicate=False, requires_flag=False)
