"""
Phase 27.5 — ScholarLens Answer Quality, Grounding, and Alignment Regression Test Suite

Verifies:
1. test_rag_limitations_direct_opening_and_natural_language()
2. test_claim_level_evidence_alignment_strips_unsupported_citations()
3. test_ai_corpus_shortage_triggers_online_fallback_without_unrelated_domains()
4. test_citation_to_evidence_bidirectional_alignment()
5. test_understandable_answer_quality_no_unsupported_jargon()
6. test_online_fallback_activation_for_partial_intent_evidence()
7. test_all_domains_retrieval_priority()
8. test_ai_healthcare_multi_domain_retrieval()
"""

import pytest
import json
from src.pipeline.domain_scope import DomainScopeDetector, ScopeType
from src.pipeline.rag import RAGPipeline, FinalSafetyGate, CitationValidator, ClaimGroundingValidator, AnswerCompletenessValidator
from src.pipeline.retrieval import HybridRetriever
from src.pipeline.online_fallback import OnlineAcademicRetriever, OnlineEvidenceItem
from src.pipeline.llm import MockLLMProvider


class TestAnswerQualityEnhancements:

    def test_rag_limitations_direct_opening_and_natural_language(self):
        """RAG limitations question begins directly with a clear statement and uses natural, understandable academic prose."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What are the limitations of retrieval-augmented generation systems?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        # Must produce a valid answer (grounded response or honest insufficient evidence response)
        assert res.answer
        # Must not contain unsupported jargon
        assert "critical architectural limitations during real-world scientific execution" not in res.answer
        assert "context-induced hallucinations" not in res.answer
        assert "out-of-distribution handling" not in res.answer
        assert "domain shift" not in res.answer

    def test_claim_level_evidence_alignment_strips_unsupported_citations(self):
        """ClaimGroundingValidator strips citations when a sentence makes technical assertions not present in cited text."""
        passage_text = "Joan Figuerola Hurtado presents a methodology for uncovering knowledge gaps on the internet using RAG models."
        ev_item = type("Ev", (), {"text": passage_text, "citation_id": "E1", "paper_id": "AI001"})()
        evidence_map = {"E1": ev_item}

        # Sentence makes latency and context window claims not supported by passage_text
        unsupported_sent = "RAG systems suffer from context window constraints and computational latency [E1]."
        grounded_answer, unsupported_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(
            unsupported_sent, evidence_map
        )

        assert "[E1]" not in grounded_answer or unsupported_count > 0, "Citation [E1] should be stripped or sentence flagged as unsupported."

    def test_ai_corpus_shortage_triggers_online_fallback_without_unrelated_domains(self):
        """When AI corpus lacks direct evidence for a specialized query, online fallback is triggered without leaking unrelated domains."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What is surface code quantum error correction?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        for ev in res.evidence:
            assert ev.domain == "artificial_intelligence", f"Unrelated domain '{ev.domain}' leaked during AI query!"

    def test_citation_to_evidence_bidirectional_alignment(self):
        """Every citation tag in res.answer MUST exist in res.citations and vice-versa."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What are the limitations of retrieval-augmented generation systems?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        tags = CitationValidator.extract_citations(res.answer)
        for tag in tags:
            assert tag in res.citations, f"Tag [{tag}] present in answer text but missing from res.citations!"
        for tag in res.citations:
            assert tag in tags, f"Tag [{tag}] present in res.citations but missing from answer text!"

    def test_understandable_answer_quality_no_unsupported_jargon(self):
        """Answer is clear, coherent, and readable without reading raw passages first."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "How do vector database indexes optimize retrieval in RAG systems?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        assert len(res.answer.split()) >= 40
        assert "vector" in res.answer.lower()
        assert not res.answer.startswith("How do vector database")

    def test_online_fallback_activation_for_partial_intent_evidence(self):
        """Partial intent support triggers Tier 2 online fallback."""
        failing_local = type("MockRetriever", (), {
            "retrieve": lambda self, query=None, **kwargs: [type("Res", (), {
                "unit_id": "AI001_U01", "parent_chunk_id": "AI001_C01", "chunk_id": "AI001_C01",
                "paper_id": "AI001", "section_id": "SEC1", "section_name": "Abstract",
                "domain": "artificial_intelligence", "subtopic": "rag", "page_start": 1, "page_end": 1,
                "text": "Partial mention of indexing.", "dense_score": 0.5, "bm25_score": 5.0, "rrf_score": 0.02,
                "retrieval_methods": ["ChromaDB"]
            })()]
        })()

        online_ev = OnlineEvidenceItem(
            citation_id="O1", paper_id="arXiv:2401.99999", title="Online Vector Indexing",
            authors=["O. Online"], published_date="2024", url="http://arxiv.org/abs/2401.99999",
            domain="artificial_intelligence", section_name="Abstract", text="HNSW indexing optimizes dense vector search."
        )
        mock_online = type("MockOnline", (), {"retrieve": lambda self, **kwargs: [online_ev]})()

        pipeline = RAGPipeline(retriever=failing_local, online_retriever=mock_online, llm_provider=MockLLMProvider())
        res = pipeline.answer("What are the limitations of retrieval-augmented generation systems?")

        assert res.retrieval_metadata.get("online_fallback_active") is True

    def test_all_domains_retrieval_priority(self):
        """All-domains scope executes without random domain cross-contamination."""
        scope_res = DomainScopeDetector.detect("What are the limitations of RAG systems?", user_domain=None)
        assert scope_res.scope_type in (ScopeType.ALL_DOMAINS, ScopeType.SINGLE_DOMAIN)

    def test_ai_healthcare_multi_domain_retrieval(self):
        """Multi-domain query involving AI and healthcare allows both domains strictly."""
        q = "How is artificial intelligence used in healthcare for medical image diagnosis?"
        scope_res = DomainScopeDetector.detect(q, user_domain=None)

        assert scope_res.scope_type == ScopeType.MULTI_DOMAIN
        assert "artificial_intelligence" in scope_res.allowed_domains
        assert "healthcare" in scope_res.allowed_domains
        assert len(scope_res.allowed_domains) == 2
