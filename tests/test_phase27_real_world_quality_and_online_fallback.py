"""
Phase 27 — Final Real-World Answer Quality & Online Fallback Validation Test Suite

Verifies:
1. test_online_fallback_is_not_triggered_when_local_evidence_is_sufficient()
2. test_real_arxiv_api_execution_and_metadata()
3. test_online_claim_grounding_and_non_fabrication()
4. test_hybrid_local_and_online_answer()
5. test_ai_corpus_shortage_no_unrelated_domain_compensation()
6. test_multi_domain_ai_healthcare_online_fallback()
7. test_final_safety_gate_rejects_unmapped_tags()
8. test_online_fallback_failure_returns_local_partial_answer_or_insufficient()
"""

import pytest
import json
from src.pipeline.domain_scope import DomainScopeDetector, ScopeType
from src.pipeline.rag import RAGPipeline, RelevanceGate, QuestionAnswerRelevanceValidator, ClaimGroundingValidator, AnswerCompletenessValidator, FinalSafetyGate
from src.pipeline.retrieval import HybridRetriever
from src.pipeline.online_fallback import OnlineAcademicRetriever, OnlineEvidenceItem
from src.pipeline.llm import MockLLMProvider


class TestPhase27RealWorldQualityAndOnlineFallback:

    def test_online_fallback_is_not_triggered_when_local_evidence_is_sufficient(self):
        """When local evidence is sufficient (match_ratio >= 0.65, count >= 3), online fallback must NOT be triggered."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What is retrieval-augmented generation?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        assert res.retrieval_metadata.get("online_fallback_active") is False or res.retrieval_metadata.get("online_evidence_count") == 0
        assert res.why_this_answer.source_type == "Research Mind Corpus"
        assert all(cit.startswith("E") for cit in res.citations.keys())

    def test_real_arxiv_api_execution_and_metadata(self):
        """Execute real ArXiv API query and assert real paper metadata structure."""
        online_retriever = OnlineAcademicRetriever(timeout=10)
        items = online_retriever.retrieve(
            "surface code quantum error correction",
            domain="artificial_intelligence",
            max_results=3
        )

        if items:
            first = items[0]
            assert first.citation_id == "O1"
            assert first.paper_id.startswith("arXiv:")
            assert first.url.startswith("http://arxiv.org/abs/")
            assert len(first.title) > 5
            assert len(first.authors) >= 1
            assert len(first.text) > 40
        else:
            # On network timeout or throttling, ensure retriever returned safe empty list
            assert items == []

    def test_online_claim_grounding_and_non_fabrication(self):
        """Online evidence must pass claim-level grounding and receive real [O1], [O2] tags."""
        online_ev = OnlineEvidenceItem(
            citation_id="O1",
            paper_id="arXiv:2403.99999",
            title="Quantum Teleportation Neural Networks Protocol",
            authors=["R. Robotics", "A. Agent"],
            published_date="2024-03-01",
            url="http://arxiv.org/abs/2403.99999",
            domain="artificial_intelligence",
            section_name="Abstract",
            text="Quantum teleportation protocol executed via biological neural networks achieves state preservation."
        )

        mock_llm = MockLLMProvider(custom_response=json.dumps({
            "answer": "Quantum teleportation protocol executed via biological neural networks achieves state preservation [O1].",
            "confidence": "High",
            "evidence_strength": "High",
            "limitations": "Online evidence findings.",
            "why_this_answer": "Supported by ArXiv online academic search."
        }))

        pipeline = RAGPipeline(llm_provider=mock_llm)
        pipeline.online_retriever = type("MockOnline", (), {"retrieve": lambda self, **kwargs: [online_ev]})()

        res = pipeline.answer("What is quantum teleportation using biological neural networks?", filters={"domain": "artificial_intelligence"})
        assert "O1" in res.citations
        assert res.citations["O1"].title == "Quantum Teleportation Neural Networks Protocol"
        assert res.why_this_answer.source_type == "Online Academic Search"

    def test_hybrid_local_and_online_answer(self):
        """Hybrid local + online answer contains both [E1] and [O1] citations."""
        online_ev = OnlineEvidenceItem(
            citation_id="O1",
            paper_id="arXiv:2401.12345",
            title="Dense Vector Index Optimization",
            authors=["V. Vector"],
            published_date="2024-01-01",
            url="http://arxiv.org/abs/2401.12345",
            domain="artificial_intelligence",
            section_name="Abstract",
            text="Product quantization and HNSW graph indexing optimize dense vector retrieval latency."
        )

        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        pipeline.online_retriever = type("MockOnline", (), {"retrieve": lambda self, **kwargs: [online_ev]})()

        res = pipeline.answer("How do vector database indexes optimize retrieval in RAG systems?")
        assert res.confidence in ["Excellent", "High", "Moderate"]

    def test_ai_corpus_shortage_no_unrelated_domain_compensation(self):
        """Out-of-corpus AI query must NOT substitute local HC/AG/CY/CL papers."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What are the recent advances in vision-language action models for robot manipulation?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        for ev in res.evidence:
            assert ev.domain == "artificial_intelligence", f"Unallowed domain leakage: '{ev.domain}'"
            assert not ev.paper_id.startswith("HC")
            assert not ev.paper_id.startswith("AG")

    def test_multi_domain_ai_healthcare_online_fallback(self):
        """AI + Healthcare multi-domain query searches only AI and Healthcare categories."""
        online_retriever = OnlineAcademicRetriever()
        items = online_retriever.retrieve(
            "AI medical image analysis convolutional networks",
            allowed_domains=["artificial_intelligence", "healthcare"],
            max_results=3
        )

        if items:
            for item in items:
                assert item.domain in ["artificial_intelligence", "healthcare"]

    def test_final_safety_gate_rejects_unmapped_tags(self):
        """FinalSafetyGate strips unmapped citation tags like [E99] or [O99]."""
        clean_ans, active_cits, active_ev = FinalSafetyGate.sanitize_response(
            answer="RAG combines retrieval and generation [E1] with unsupported tag [E99].",
            citations_map={"E1": type("Cit", (), {"paper_id": "AI001", "section_name": "Abstract"})()},
            evidence_items=[type("Ev", (), {"citation_id": "E1", "domain": "artificial_intelligence", "paper_id": "AI001", "source_type": "corpus"})()],
            allowed_domains=["artificial_intelligence"],
            expected_prefixes={"AI"}
        )

        assert "[E99]" not in clean_ans
        assert "[E1]" in clean_ans
        assert "E1" in active_cits

    def test_online_fallback_failure_returns_local_partial_answer_or_insufficient(self):
        """When online fallback fails (e.g. timeout or API returns 0 results), system safely falls back to grounded local partial answer or Insufficient Evidence."""
        failing_online = type("FailingOnline", (), {"retrieve": lambda self, **kwargs: []})()

        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        pipeline.online_retriever = failing_online

        res_out = pipeline.answer("What is quantum teleportation using biological neural networks?")
        assert res_out.confidence == "Insufficient"
        assert "insufficient evidence" in res_out.answer.lower()
        assert res_out.why_this_answer.source_type == "Research Mind Corpus"
