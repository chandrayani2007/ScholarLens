"""
Phase 5 — RAG Pipeline Integration & Provenance Verification Test Suite
"""

import pytest
from src.pipeline.rag import (
    RAGPipeline,
    EvidenceContextBuilder,
    CitationValidator,
    RelevanceGate,
    EvidenceItem,
    Citation,
    WhyThisAnswer,
    RAGResponse,
)
from src.pipeline.retrieval import RetrievalResult
from src.pipeline.llm import MockLLMProvider, get_llm_provider


@pytest.fixture
def sample_retrieval_results():
    return [
        RetrievalResult(
            rank=1,
            unit_id="AI003_C001",
            chunk_id="AI003_C001",
            parent_chunk_id="AI003_C001",
            paper_id="AI003",
            section_id="SEC_001",
            section_name="Introduction",
            domain="artificial_intelligence",
            subtopic="rag",
            page_start=1,
            page_end=2,
            text="Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with LLMs.",
            token_count=120,
            dense_score=0.85,
            bm25_score=12.4,
            rrf_score=0.032,
            retrieval_methods=["both"],
        ),
        RetrievalResult(
            rank=2,
            unit_id="AI012_C004",
            chunk_id="AI012_C004",
            parent_chunk_id="AI012_C004",
            paper_id="AI012",
            section_id="SEC_002",
            section_name="Architecture",
            domain="artificial_intelligence",
            subtopic="rag",
            page_start=3,
            page_end=4,
            text="The retriever fetches top-k passages from ChromaDB and BM25 indexers before rank fusion.",
            token_count=110,
            dense_score=0.78,
            bm25_score=9.1,
            rrf_score=0.028,
            retrieval_methods=["ChromaDB"],
        ),
    ]


class TestEvidenceContextBuilder:
    def test_build_context_formatting(self, sample_retrieval_results):
        context_text, evidence_items, citations_map = EvidenceContextBuilder.build_context(sample_retrieval_results)

        assert "[E1]" in context_text
        assert "[E2]" in context_text
        assert "Paper: AI003" in context_text
        assert "Section: Introduction" in context_text
        assert "Pages: 1-2" in context_text

    def test_provenance_preservation(self, sample_retrieval_results):
        _, evidence_items, _ = EvidenceContextBuilder.build_context(sample_retrieval_results)

        assert len(evidence_items) == 2
        ev1 = evidence_items[0]
        assert isinstance(ev1, EvidenceItem)
        assert ev1.citation_id == "E1"
        assert ev1.paper_id == "AI003"
        assert ev1.section_name == "Introduction"
        assert ev1.page_start == 1
        assert ev1.page_end == 2
        assert ev1.rrf_score == 0.032

    def test_citation_mapping(self, sample_retrieval_results):
        _, _, citations_map = EvidenceContextBuilder.build_context(sample_retrieval_results)

        assert "E1" in citations_map
        cit1 = citations_map["E1"]
        assert isinstance(cit1, Citation)
        assert cit1.paper_id == "AI003"
        assert cit1.section_name == "Introduction"
        assert cit1.pages == "1-2"


class TestCitationValidator:
    def test_extract_citations(self):
        text = "RAG combines vector search [E1] with BM25 keyword search [E2, E3]."
        extracted = CitationValidator.extract_citations(text)
        assert extracted == ["E1", "E2", "E3"]

    def test_valid_citation_validation(self, sample_retrieval_results):
        _, _, citations_map = EvidenceContextBuilder.build_context(sample_retrieval_results)
        text = "RAG framework [E1] uses hybrid search [E2]."

        is_valid, valid, invalid = CitationValidator.validate(text, citations_map)
        assert is_valid is True
        assert valid == ["E1", "E2"]
        assert invalid == []

    def test_invalid_citation_detection(self, sample_retrieval_results):
        _, _, citations_map = EvidenceContextBuilder.build_context(sample_retrieval_results)
        text = "RAG framework [E1] uses hallucinated source [E99]."

        is_valid, valid, invalid = CitationValidator.validate(text, citations_map)
        assert is_valid is False
        assert valid == ["E1"]
        assert invalid == ["E99"]


class TestRAGPipeline:
    def test_mock_llm_provider(self):
        llm = get_llm_provider("mock")
        res = llm.generate("Research Question: What is RAG?\n[E1]\nText:\nRetrieval-Augmented Generation (RAG) is a technique for grounding LLMs.")
        assert "RAG" in res or "Retrieval-Augmented Generation" in res
        assert "[E1]" in res

    def test_rag_pipeline_end_to_end(self):
        pipeline = RAGPipeline()
        response = pipeline.answer("What is retrieval augmented generation?")

        assert isinstance(response, RAGResponse)
        assert response.question == "What is retrieval augmented generation?"
        assert len(response.answer) > 0
        assert isinstance(response.why_this_answer, WhyThisAnswer)
        assert response.confidence in {"Excellent", "High", "Medium", "Low", "Insufficient"}
        assert len(response.evidence) > 0

    def test_insufficient_evidence_fallback(self):
        mock_llm = MockLLMProvider(custom_response='{"answer": "Insufficient evidence was found in the current Research Mind corpus to answer this question reliably."}')
        pipeline = RAGPipeline(llm_provider=mock_llm)
        response = pipeline.answer("What is the exact secret password of Atlantis?")

        assert "Insufficient evidence" in response.answer
        assert response.confidence in {"Low", "Insufficient"}
        assert response.why_this_answer.evidence_strength in {"Weak", "Insufficient"}
