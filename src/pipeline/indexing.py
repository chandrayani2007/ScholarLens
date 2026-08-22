"""
Phase 4 — BM25 Lexical Indexer with Metadata Filter Support

Manages:
1. Tokenization and BM25Okapi inverted index over 13,718 text units.
2. In-memory serialization via pickle to data/indexes/bm25.pkl.
3. Lexical search execution with metadata pre-filtering (domain, subtopic, paper_id).
"""

import pickle
import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import numpy as np
from rank_bm25 import BM25Okapi

from src.pipeline.embedding import IndexingConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def tokenize_text(text: str) -> List[str]:
    """Tokenize input text for BM25 indexing and searching."""
    if not text:
        return []
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


class BM25Indexer:
    """
    BM25Okapi lexical index manager supporting unit searching and metadata filtering.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.units: List[Dict[str, Any]] = []

    def build_index(self, units: List[Dict[str, Any]]) -> None:
        """
        Tokenize and index text units using BM25Okapi.
        """
        if not units:
            raise ValueError("Cannot build BM25 index on empty units list")

        logger.info(f"Tokenizing {len(units)} units for BM25 indexing...")
        corpus = [tokenize_text(u["text"]) for u in units]
        self.bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        self.units = units
        logger.info(f"Successfully built BM25 index over {len(self.units)} units")

    def save(self, index_path: Path) -> None:
        """Serialize BM25 index and unit metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "bm25": self.bm25,
            "units": self.units,
            "k1": self.k1,
            "b": self.b,
        }
        with open(index_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved BM25 index to {index_path}")

    def load(self, index_path: Path) -> None:
        """Load BM25 index and unit metadata from disk."""
        if not index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found at {index_path}")
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.units = data["units"]
        self.k1 = data.get("k1", 1.5)
        self.b = data.get("b", 0.75)
        logger.info(f"Loaded BM25 index over {len(self.units)} units")

    def search(
        self,
        query_str: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search BM25 index with query string and optional metadata filters.
        Returns top k matched result dicts with BM25 scores and provenance.
        """
        if self.bm25 is None or not self.units:
            return []

        tokenized_query = tokenize_text(query_str)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        if filters:
            filtered_indices = []
            for idx, u in enumerate(self.units):
                match = True
                for fk, fv in filters.items():
                    if fv is not None:
                        item_val = u.get(fk)
                        if isinstance(fv, (list, set, tuple)):
                            if item_val not in fv:
                                match = False
                                break
                        else:
                            if item_val != fv:
                                match = False
                                break
                if match:
                    filtered_indices.append(idx)

            if not filtered_indices:
                return []

            filtered_arr = np.array(filtered_indices)
            filtered_scores = scores[filtered_arr]
            top_sub_indices = np.argsort(filtered_scores)[::-1][:k]
            top_indices = filtered_arr[top_sub_indices]
        else:
            top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0 and len(results) >= k:
                break
            item = dict(self.units[idx])
            item["score"] = float(score)
            item["bm25_index"] = int(idx)
            results.append(item)

        return results


def write_index_metadata(
    metadata_path: Path,
    config: IndexingConfig,
    total_original_chunks: int,
    total_embedding_units: int,
    dense_vector_count: int,
    bm25_doc_count: int,
    extra: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create reproducible index_metadata.json artifact."""
    metadata = {
        "dataset_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunking_summary": {
            "total_original_chunks": total_original_chunks,
            "total_embedding_units": total_embedding_units,
            "oversized_sub_chunks_added": total_embedding_units - total_original_chunks,
        },
        "embedding_config": {
            "model_name": config.embedding_model,
            "embedding_dim": config.embedding_dim,
            "max_seq_length": config.max_seq_length,
            "query_instruction": config.query_instruction,
            "document_prefix": config.document_prefix,
            "sub_chunk_max_tokens": config.sub_chunk_max_tokens,
            "sub_chunk_overlap": config.sub_chunk_overlap,
            "normalize_embeddings": config.normalize_embeddings,
        },
        "dense_config": {
            "backend": "chromadb",
            "similarity_metric": config.similarity_metric,
            "vector_count": dense_vector_count,
        },
        "bm25_config": {
            "algorithm": "BM25Okapi",
            "k1": config.bm25_k1,
            "b": config.bm25_b,
            "document_count": bm25_doc_count,
        },
    }
    if extra:
        metadata["extra"] = extra

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved index metadata to {metadata_path}")
    return metadata
