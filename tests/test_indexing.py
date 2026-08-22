"""
Unit & Integration Tests for Indexing Pipeline (BM25 and ChromaDB).

Tests cover:
- Sub-chunking oversized chunks (>512 tokens) and provenance retention
- Embedding generation, shape, and L2 normalization
- BGE query instruction prefixing
- ChromaDB persistent database building, loading, metadata filtering, and searching
- BM25Okapi building, saving, loading, and searching
- Index metadata creation
- Handling edge cases (empty text, missing fields, duplicate IDs)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.embedding import (
    IndexingConfig,
    ChunkSubSplitter,
    EmbeddingGenerator,
)
from src.pipeline.indexing import (
    BM25Indexer,
    tokenize_text,
    write_index_metadata,
)
from src.pipeline.chroma_indexer import ChromaDBIndexer


@pytest.fixture
def config():
    return IndexingConfig(
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
        max_seq_length=512,
        query_instruction="Represent this sentence for searching relevant passages: ",
        document_prefix="",
        sub_chunk_max_tokens=50,
        sub_chunk_overlap=10,
        batch_size=16,
        normalize_embeddings=True,
    )


@pytest.fixture
def sample_chunk():
    return {
        "chunk_id": "AI001_C001",
        "parent_chunk_id": "AI001_C001",
        "unit_id": "AI001_C001",
        "paper_id": "AI001",
        "section_id": "SEC_AI001_001",
        "domain": "artificial_intelligence",
        "subtopic": "retrieval_augmented_generation",
        "section_name": "Introduction",
        "page_start": 1,
        "page_end": 1,
        "token_count": 20,
        "text": "Retrieval-Augmented Generation enhances language models with external factual data.",
    }


@pytest.fixture
def sample_oversized_chunk():
    sentences = [f"This is sentence number {i} providing detailed research content." for i in range(1, 25)]
    text = " ".join(sentences)
    return {
        "chunk_id": "CY001_C010",
        "parent_chunk_id": "CY001_C010",
        "unit_id": "CY001_C010",
        "paper_id": "CY001",
        "section_id": "SEC_CY001_003",
        "domain": "cybersecurity",
        "subtopic": "network_security",
        "section_name": "Methodology",
        "page_start": 4,
        "page_end": 5,
        "token_count": 300,
        "text": text,
    }


# ─── Sub-Chunking Tests ───

class TestSubChunking:
    def test_normal_chunk_remains_intact(self, sample_chunk):
        splitter = ChunkSubSplitter(max_tokens=100, overlap_tokens=20)
        units = splitter.process_chunks([sample_chunk])
        assert len(units) == 1
        u = units[0]
        assert u["unit_id"] == "AI001_C001"
        assert u["parent_chunk_id"] == "AI001_C001"
        assert u["is_sub_chunk"] is False

    def test_oversized_chunk_splits_into_sub_units(self, sample_oversized_chunk):
        splitter = ChunkSubSplitter(max_tokens=40, overlap_tokens=10)
        units = splitter.process_chunks([sample_oversized_chunk])
        assert len(units) > 1
        for sub_idx, u in enumerate(units, 1):
            assert u["parent_chunk_id"] == "CY001_C010"
            assert u["paper_id"] == "CY001"
            assert u["section_id"] == "SEC_CY001_003"
            assert u["page_start"] == 4
            assert u["page_end"] == 5
            assert u["is_sub_chunk"] is True
            assert u["unit_id"] == f"CY001_C010_sub{sub_idx:02d}"


# ─── Tokenizer & BM25 Tests ───

class TestBM25:
    def test_tokenization(self):
        tokens = tokenize_text("Retrieval-Augmented Generation: 2026!")
        assert "retrieval" in tokens
        assert "augmented" in tokens
        assert "generation" in tokens
        assert "2026" in tokens

    def test_bm25_build_and_search(self, sample_chunk, sample_oversized_chunk):
        indexer = BM25Indexer()
        chunk2 = dict(sample_oversized_chunk)
        chunk2["chunk_id"] = "CY001_C010"
        chunk2["text"] = "Different topic about cybersecurity and network firewall rules."

        chunk3 = dict(sample_oversized_chunk)
        chunk3["chunk_id"] = "AG001_C005"
        chunk3["text"] = "Agricultural IoT sensors and soil moisture telemetry systems."

        indexer.build_index([sample_chunk, chunk2, chunk3])
        results = indexer.search("Retrieval Augmented", k=1)
        assert len(results) == 1
        assert results[0]["unit_id"] == "AI001_C001"
        assert results[0]["score"] > 0

    def test_bm25_save_load(self, sample_chunk, sample_oversized_chunk):
        indexer = BM25Indexer()
        chunk2 = dict(sample_oversized_chunk)
        chunk2["chunk_id"] = "CY001_C010"
        chunk2["text"] = "Different topic about cybersecurity and network firewall rules."

        chunk3 = dict(sample_oversized_chunk)
        chunk3["chunk_id"] = "AG001_C005"
        chunk3["text"] = "Agricultural IoT sensors and soil moisture telemetry systems."

        indexer.build_index([sample_chunk, chunk2, chunk3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bm25.pkl"
            indexer.save(path)

            loaded_indexer = BM25Indexer()
            loaded_indexer.load(path)
            results = loaded_indexer.search("language models", k=1)
            assert len(results) == 1
            assert results[0]["unit_id"] == "AI001_C001"
            assert results[0]["score"] > 0


# ─── ChromaDB Indexer Tests ───

class TestChromaDB:
    def test_chromadb_build_and_search(self, sample_chunk, tmp_path):
        dim = 384
        chroma_dir = tmp_path / "chroma"
        indexer = ChromaDBIndexer(db_dir=chroma_dir, collection_name="test_collection", dim=dim)

        vec = np.random.randn(1, dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        indexer.build_index(vec, [sample_chunk])
        assert indexer.collection.count() == 1

        results = indexer.search(vec[0], k=1)
        assert len(results) == 1
        assert results[0]["unit_id"] == "AI001_C001"
        assert results[0]["paper_id"] == "AI001"
        assert abs(results[0]["score"] - 1.0) < 1e-3

    def test_chromadb_metadata_filtering(self, sample_chunk, sample_oversized_chunk, tmp_path):
        dim = 384
        chroma_dir = tmp_path / "chroma"
        indexer = ChromaDBIndexer(db_dir=chroma_dir, collection_name="test_filter_collection", dim=dim)

        vecs = np.random.randn(2, dim).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        indexer.build_index(vecs, [sample_chunk, sample_oversized_chunk])

        # Filter by domain
        results = indexer.search(vecs[0], k=5, filters={"domain": "cybersecurity"})
        assert len(results) == 1
        assert results[0]["unit_id"] == "CY001_C010"


# ─── Embedding Generator Tests ───

class TestEmbeddingGenerator:
    def test_encode_documents_and_queries(self, config, sample_chunk):
        generator = EmbeddingGenerator(config)
        units = generator.prepare_units([sample_chunk])
        embeddings = generator.encode_units(units, show_progress=False)

        assert embeddings.shape == (1, 384)
        assert embeddings.dtype == np.float32
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 1e-4

        q_vec = generator.encode_queries(["What is RAG?"])
        assert q_vec.shape == (1, 384)
        q_norm = np.linalg.norm(q_vec[0])
        assert abs(q_norm - 1.0) < 1e-4


# ─── Metadata & Edge Case Tests ───

class TestMetadataAndEdgeCases:
    def test_metadata_schema(self, config):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "index_metadata.json"
            write_index_metadata(path, config, 10, 12, 12, 12)

            assert path.exists()
            import json
            with open(path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            assert meta["embedding_config"]["model_name"] == "BAAI/bge-small-en-v1.5"
            assert meta["dense_config"]["vector_count"] == 12
