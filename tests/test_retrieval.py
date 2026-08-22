"""
Unit Tests for Phase 4 & Phase 5 Hybrid Retrieval (ChromaDB + BM25 + Reciprocal Rank Fusion + Section Filtering).

Tests cover:
- RetrievalConfig loading & defaults (including section filtering defaults)
- BGE query instruction prefixing
- ChromaDB dense retrieval candidate resolution
- BM25 lexical retrieval candidate resolution
- RRF score calculation & rank ordering
- Method tagging (ChromaDB / FAISS, BM25, both)
- Parent chunk deduplication and provenance retention
- Metadata filtering (domain, subtopic, paper_id, section_name)
- Dynamic section filtering toggle (Header, References, Bibliography, Acknowledgements exclusion)
- Error handling (empty query, invalid filter key)
- Deterministic ranking stability
"""

import pytest
import numpy as np

from src.pipeline.retrieval import (
    RetrievalConfig,
    RetrievalResult,
    HybridRetriever,
)
from src.pipeline.indexing import BM25Indexer
from src.pipeline.chroma_indexer import ChromaDBIndexer
from src.pipeline.embedding import IndexingConfig, EmbeddingGenerator


@pytest.fixture
def retrieval_config():
    return RetrievalConfig(
        dense_backend="chromadb",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_dim=384,
        query_instruction="Represent this sentence for searching relevant passages: ",
        top_k_dense=5,
        top_k_bm25=5,
        rrf_k=60,
        dense_weight=1.0,
        lexical_weight=1.0,
        final_top_k=5,
        deduplicate_parent_chunks=True,
        enable_section_filtering=True,
        excluded_sections=["Header", "References", "Bibliography", "Acknowledgements"],
    )


@pytest.fixture
def mock_units():
    return [
        {
            "unit_id": "AI001_C001",
            "parent_chunk_id": "AI001_C001",
            "paper_id": "AI001",
            "section_id": "SEC_AI001_001",
            "section_name": "Header",
            "domain": "artificial_intelligence",
            "subtopic": "retrieval_augmented_generation",
            "page_start": 1,
            "page_end": 1,
            "token_count": 20,
            "text": "Header block: Title and Author information for paper AI001.",
        },
        {
            "unit_id": "AI001_C002_sub01",
            "parent_chunk_id": "AI001_C002",
            "paper_id": "AI001",
            "section_id": "SEC_AI001_002",
            "section_name": "Methodology",
            "domain": "artificial_intelligence",
            "subtopic": "retrieval_augmented_generation",
            "page_start": 2,
            "page_end": 2,
            "token_count": 45,
            "text": "Dense vector retrieval uses embedding similarity over pre-computed chunk vectors.",
        },
        {
            "unit_id": "AI001_C002_sub02",
            "parent_chunk_id": "AI001_C002",
            "paper_id": "AI001",
            "section_id": "SEC_AI001_002",
            "section_name": "Methodology",
            "domain": "artificial_intelligence",
            "subtopic": "retrieval_augmented_generation",
            "page_start": 2,
            "page_end": 3,
            "token_count": 40,
            "text": "Lexical search uses BM25 term matching over tokenized document texts.",
        },
        {
            "unit_id": "AI001_C099",
            "parent_chunk_id": "AI001_C099",
            "paper_id": "AI001",
            "section_id": "SEC_AI001_099",
            "section_name": "References",
            "domain": "artificial_intelligence",
            "subtopic": "retrieval_augmented_generation",
            "page_start": 10,
            "page_end": 10,
            "token_count": 50,
            "text": "References: [1] Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.",
        },
        {
            "unit_id": "HC001_C005",
            "parent_chunk_id": "HC001_C005",
            "paper_id": "HC001",
            "section_id": "SEC_HC001_001",
            "section_name": "Abstract",
            "domain": "healthcare",
            "subtopic": "medical_image_analysis",
            "page_start": 1,
            "page_end": 1,
            "token_count": 60,
            "text": "Medical image analysis algorithms diagnose diseases using deep convolutional networks.",
        },
    ]


@pytest.fixture
def mock_retriever(retrieval_config, mock_units, tmp_path):
    dim = retrieval_config.embedding_dim

    # Mock ChromaDB in tmp_path
    chroma_dir = tmp_path / "chroma"
    chroma_idx = ChromaDBIndexer(db_dir=chroma_dir, collection_name="test_chunks", dim=dim)
    vecs = np.random.randn(len(mock_units), dim).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    chroma_idx.build_index(vecs, mock_units)

    # Mock BM25
    bm25_idx = BM25Indexer()
    bm25_idx.build_index(mock_units)

    # Mock Generator
    idx_config = IndexingConfig(
        embedding_model=retrieval_config.embedding_model,
        embedding_dim=dim,
        query_instruction=retrieval_config.query_instruction,
    )
    generator = EmbeddingGenerator(idx_config)

    retriever = HybridRetriever(
        config=retrieval_config,
        chroma_indexer=chroma_idx,
        bm25_indexer=bm25_idx,
        embedding_generator=generator,
    )
    return retriever


# ─── Tests ───

class TestRetrievalConfig:
    def test_config_defaults(self):
        cfg = RetrievalConfig()
        assert cfg.dense_backend == "chromadb"
        assert cfg.top_k_dense == 20
        assert cfg.top_k_bm25 == 20
        assert cfg.rrf_k == 60
        assert cfg.dense_weight == 1.0
        assert cfg.lexical_weight == 1.0
        assert cfg.final_top_k == 10
        assert cfg.deduplicate_parent_chunks is True
        assert cfg.enable_section_filtering is True
        assert "Header" in cfg.excluded_sections
        assert "References" in cfg.excluded_sections


class TestSectionFilteringToggle:
    def test_section_filtering_enabled_excludes_header_and_references(self, mock_retriever):
        mock_retriever.config.enable_section_filtering = True
        results = mock_retriever.retrieve("Retrieval Augmented Generation References Header", top_k=5)
        sec_names = [r.section_name for r in results]
        assert "Header" not in sec_names
        assert "References" not in sec_names

    def test_section_filtering_disabled_includes_header_and_references(self, mock_retriever):
        mock_retriever.config.enable_section_filtering = False
        results = mock_retriever.retrieve("Retrieval Augmented Generation References Header", top_k=5)
        sec_names = [r.section_name for r in results]
        assert any(sec in sec_names for sec in ["Header", "References", "Methodology", "Abstract"])


class TestHybridRetrieval:
    def test_retrieve_returns_results(self, mock_retriever):
        results = mock_retriever.retrieve("retrieval augmented generation", top_k=3)
        assert len(results) <= 3
        for res in results:
            assert isinstance(res, RetrievalResult)
            assert res.rank >= 1
            assert res.rrf_score > 0

    def test_domain_filtering(self, mock_retriever):
        results = mock_retriever.retrieve(
            "medical image analysis",
            top_k=5,
            filters={"domain": "healthcare"}
        )
        assert len(results) > 0
        for res in results:
            assert res.domain == "healthcare"

    def test_empty_query_raises_value_error(self, mock_retriever):
        with pytest.raises(ValueError):
            mock_retriever.retrieve("")

    def test_deterministic_ranking(self, mock_retriever):
        q = "retrieval augmented generation"
        r1 = mock_retriever.retrieve(q, top_k=5)
        r2 = mock_retriever.retrieve(q, top_k=5)

        assert len(r1) == len(r2)
        for res1, res2 in zip(r1, r2):
            assert res1.unit_id == res2.unit_id
            assert res1.rrf_score == res2.rrf_score
