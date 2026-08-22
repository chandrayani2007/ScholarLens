"""
Phase 3 — Embedding Generator & Sub-Chunk Processor

Handles:
1. Oversized chunk (>512 tokens) sentence-level sub-splitting without silent truncation.
2. Provenance propagation (unit_id, parent_chunk_id, paper_id, section_id, page_start, page_end).
3. BAAI/bge-small-en-v1.5 local embedding generation via sentence-transformers.
4. BGE query instruction prefixing for search queries.
5. L2 normalization & embedding caching/resume capabilities.
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from src.pipeline.chunking import split_into_sentences

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class IndexingConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    max_seq_length: int = 512
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    document_prefix: str = ""
    sub_chunk_max_tokens: int = 480
    sub_chunk_overlap: int = 50
    batch_size: int = 64
    normalize_embeddings: bool = True
    similarity_metric: str = "cosine"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    @classmethod
    def from_file(cls, config_path: Path) -> "IndexingConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ChunkSubSplitter:
    """
    Ensures no chunk exceeds max_seq_length (512 tokens) by splitting oversized
    chunks into sub-units at sentence boundaries with overlap.
    Preserves exact parent provenance across all sub-units.
    """

    def __init__(self, max_tokens: int = 480, overlap_tokens: int = 50, encoding_name: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of chunk dicts into embedding units.
        Chunks with token_count <= max_tokens remain intact.
        Chunks with token_count > max_tokens are split into sub-chunks.
        """
        units = []
        for chunk in chunks:
            token_count = chunk.get("token_count", self._count_tokens(chunk["text"]))
            if token_count <= self.max_tokens:
                # Intact chunk
                unit = dict(chunk)
                unit["unit_id"] = chunk["chunk_id"]
                unit["parent_chunk_id"] = chunk["chunk_id"]
                unit["is_sub_chunk"] = False
                unit["sub_index"] = 0
                units.append(unit)
            else:
                # Oversized chunk: split into sub-units at sentence boundaries
                sub_units = self._split_oversized_chunk(chunk)
                units.extend(sub_units)

        logger.info(f"Sub-chunking expanded {len(chunks)} original chunks into {len(units)} embedding units")
        return units

    def _split_oversized_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        sentences = split_into_sentences(chunk["text"])
        if not sentences:
            sentences = [chunk["text"]]

        sentence_tokens = [self._count_tokens(s) for s in sentences]

        sub_units = []
        current_sentences = []
        current_tokens = 0
        sub_idx = 1

        for sent, sent_toks in zip(sentences, sentence_tokens):
            if current_sentences and (current_tokens + sent_toks > self.max_tokens):
                # Emit current accumulator
                sub_text = " ".join(current_sentences).strip()
                sub_units.append(self._create_sub_unit(chunk, sub_text, sub_idx))
                sub_idx += 1

                # Retain overlap sentences from tail
                overlap_sentences = []
                overlap_tokens = 0
                for prev_sent in reversed(current_sentences):
                    prev_toks = self._count_tokens(prev_sent)
                    if overlap_tokens + prev_toks <= self.overlap_tokens or not overlap_sentences:
                        overlap_sentences.insert(0, prev_sent)
                        overlap_tokens += prev_toks
                    else:
                        break

                current_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_sentences.append(sent)
            current_tokens += sent_toks

        # Final remaining tail
        if current_sentences:
            sub_text = " ".join(current_sentences).strip()
            sub_units.append(self._create_sub_unit(chunk, sub_text, sub_idx))

        return sub_units

    def _create_sub_unit(self, parent_chunk: Dict[str, Any], text: str, sub_idx: int) -> Dict[str, Any]:
        unit = dict(parent_chunk)
        unit["unit_id"] = f"{parent_chunk['chunk_id']}_sub{sub_idx:02d}"
        unit["parent_chunk_id"] = parent_chunk["chunk_id"]
        unit["is_sub_chunk"] = True
        unit["sub_index"] = sub_idx
        unit["text"] = text
        unit["token_count"] = self._count_tokens(text)
        return unit


class EmbeddingGenerator:
    """
    Local embedding generator using SentenceTransformer (BAAI/bge-small-en-v1.5).
    Applies query instruction prefixes to search queries and manages vector caching.
    """

    def __init__(self, config: IndexingConfig):
        self.config = config
        try:
            self.model = SentenceTransformer(config.embedding_model)
        except Exception:
            logger.info("HF Hub network connection failed, loading local cached model...")
            self.model = SentenceTransformer(config.embedding_model, local_files_only=True)
        self.model.max_seq_length = config.max_seq_length
        self.splitter = ChunkSubSplitter(
            max_tokens=config.sub_chunk_max_tokens,
            overlap_tokens=config.sub_chunk_overlap,
        )

    def prepare_units(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand chunks into embedding units (sub-splitting oversized chunks)."""
        return self.splitter.process_chunks(chunks)

    def encode_units(
        self,
        units: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a list of document embedding units.
        Returns numpy float32 array of shape (num_units, embedding_dim).
        """
        if not units:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        texts = [f"{self.config.document_prefix}{u['text']}" for u in units]

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)
        return embeddings

    def encode_queries(
        self,
        queries: List[str]
    ) -> np.ndarray:
        """
        Generate L2-normalized embeddings for search queries using BGE query instruction.
        """
        if not queries:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        prefixed_queries = [
            f"{self.config.query_instruction}{q.strip()}"
            for q in queries
        ]

        embeddings = self.model.encode(
            prefixed_queries,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float32)
        return embeddings

    @staticmethod
    def save_cache(
        embeddings: np.ndarray,
        units: List[Dict[str, Any]],
        output_dir: Path
    ) -> Tuple[Path, Path]:
        """Save vector array and unit metadata list to cache."""
        output_dir.mkdir(parents=True, exist_ok=True)
        npy_path = output_dir / "chunk_embeddings.npy"
        meta_path = output_dir / "chunk_embeddings_ids.json"

        np.save(npy_path, embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(units, f, indent=2, ensure_ascii=False)

        logger.info(f"Cached {len(units)} embeddings to {npy_path}")
        return npy_path, meta_path

    @staticmethod
    def load_cache(cache_dir: Path) -> Optional[Tuple[np.ndarray, List[Dict[str, Any]]]]:
        """Load cached vector array and unit metadata list if present."""
        npy_path = cache_dir / "chunk_embeddings.npy"
        meta_path = cache_dir / "chunk_embeddings_ids.json"

        if npy_path.exists() and meta_path.exists():
            embeddings = np.load(npy_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                units = json.load(f)
            logger.info(f"Loaded {len(units)} cached embeddings from {npy_path}")
            return embeddings, units
        return None
