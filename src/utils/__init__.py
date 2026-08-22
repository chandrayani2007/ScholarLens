"""Utility package for text normalization and deduplication signals."""
from src.utils.text import normalize_title, levenshtein_similarity
from src.utils.dedup import DeduplicationSignal, evaluate_candidate_deduplication

__all__ = ["normalize_title", "levenshtein_similarity", "DeduplicationSignal", "evaluate_candidate_deduplication"]
