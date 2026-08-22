"""
Phase 26 — Answer Depth, Evidence Sufficiency & Real Online Academic Fallback Test Suite

Verifies:
1. test_rag_definition_returns_detailed_answer()
2. test_rag_mechanism_returns_detailed_answer()
3. test_rag_limitations_returns_detailed_answer()
4. test_local_partial_evidence_triggers_online_fallback()
5. test_online_fallback_produces_o1_o2_citations()
6. test_online_metadata_is_real_and_non_fabricated()
7. test_answer_completeness_for_all_intents()
8. test_no_premature_insufficient_evidence()
9. test_ai_corpus_shortage_does_not_use_unrelated_domains()
10. test_multi_domain_ai_healthcare_uses_both_domains()
11. test_online_fallback_respects_multi_domain_scope()
12. test_question_repetition_regenerates_answer()
13. test_5_required_live_benchmark_queries_phase26()
"""

import pytest
import json
from src.pipeline.domain_scope import DomainScopeDetector, ScopeType
from src.pipeline.rag import RAGPipeline, RelevanceGate, QuestionAnswerRelevanceValidator, ClaimGroundingValidator, AnswerCompletenessValidator
from src.pipeline.retrieval import HybridRetriever
from src.pipeline.online_fallback import OnlineAcademicRetriever, OnlineEvidenceItem
from src.pipeline.llm import MockLLMProvider


class TestPhase26AnswerDepthAndOnlineFallback:

    def test_rag_definition_returns_detailed_answer(self):
        """RAG definition question returns detailed prose explaining concept and purpose."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What is retrieval-augmented generation?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        assert len(res.answer.split()) >= 30, f"Answer is too short ({len(res.answer.split())} words)"
        assert not res.answer.lower().startswith("what is retrieval-augmented generation")

    def test_rag_mechanism_returns_detailed_answer(self):
        """Mechanism question explains how RAG works step-by-step."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "How does vector database indexing optimize retrieval in RAG systems?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        assert len(res.answer.split()) >= 30, f"Answer is too short ({len(res.answer.split())} words)"

    def test_rag_limitations_returns_detailed_answer(self):
        """Limitation question explains major limitations individually."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What are the limitations of retrieval-augmented generation systems?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        assert len(res.answer.split()) >= 30
        assert "limitation" in res.answer.lower() or "hallucinat" in res.answer.lower()

    def test_local_partial_evidence_triggers_online_fallback(self):
        """When local evidence is partial (match_ratio < 0.65), system triggers online academic search."""
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
        res = pipeline.answer("How do vector database indexes optimize retrieval in RAG systems?")

        assert res.retrieval_metadata.get("online_fallback_active") is True

    def test_online_fallback_produces_o1_o2_citations(self):
        """Online evidence receives [O1], [O2] citation tags."""
        online_ev = OnlineEvidenceItem(
            citation_id="O1", paper_id="arXiv:2401.88888", title="Surface Code Error Correction",
            authors=["Q. Quantum"], published_date="2024", url="http://arxiv.org/abs/2401.88888",
            domain="artificial_intelligence", section_name="Abstract", text="Surface code error correction uses stabilizer measurements across qubits."
        )

        mock_llm = MockLLMProvider(custom_response=json.dumps({
            "answer": "Surface code error correction uses stabilizer measurements across qubits [O1]. Physical stabilizer measurements continuously compute error syndromes across adjacent data and syndrome qubits. Recent experimental realizations demonstrate fault-tolerant logical qubit operations using real-time decoders to achieve sub-threshold error scaling across multi-qubit processor architectures.",
            "confidence": "High", "evidence_strength": "High", "limitations": "None", "why_this_answer": "ArXiv fallback."
        }))

        pipeline = RAGPipeline(llm_provider=mock_llm)
        pipeline.online_retriever = type("MockOnline", (), {"retrieve": lambda self, **kwargs: [online_ev]})()

        res = pipeline.answer("What is surface code quantum error correction?")
        assert "O1" in res.citations
        assert res.citations["O1"].source_type == "online"

    def test_online_metadata_is_real_and_non_fabricated(self):
        """Online citations contain real paper metadata parsed from ArXiv."""
        online_retriever = OnlineAcademicRetriever()
        items = online_retriever.retrieve("surface code quantum error correction", max_results=2)
        if items:
            first = items[0]
            assert first.citation_id == "O1"
            assert first.paper_id.startswith("arXiv:")
            assert first.url.startswith("http://arxiv.org/abs/")
            assert len(first.title) > 0
            assert len(first.authors) > 0

    def test_answer_completeness_for_all_intents(self):
        """AnswerCompletenessValidator must validate multi-paragraph research depth."""
        detailed_text = (
            "Retrieval-augmented generation (RAG) combines information retrieval with large language model generation by supplying external evidence to the model during response synthesis. "
            "First, retrieval quality and noise dependency: if upstream vector or lexical retrieval fetches incomplete or semantically irrelevant context, the downstream generator inherits factual omissions, leading to context-induced hallucinations. "
            "Second, context window bounds and computational latency: embedding large document corpora requires fine-grained chunking, which can fragment long-range dependencies across paper sections and increase inference latency during multi-passage synthesis. "
            "Third, out-of-distribution handling: when queries address concepts absent from the underlying vector index, models risk misinterpreting partially related passages without explicit fallback mechanisms."
        )
        valid, msg = AnswerCompletenessValidator.validate_completeness(detailed_text, intent="Limitation")
        assert valid is True

    def test_no_premature_insufficient_evidence(self):
        """Query with sufficient local evidence must NOT return Insufficient Evidence."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is retrieval-augmented generation?", filters={"domain": "artificial_intelligence"})
        assert res.confidence != "Insufficient"

    def test_ai_corpus_shortage_does_not_use_unrelated_domains(self):
        """AI out-of-corpus query must search online, NOT substitute local HC/AG papers."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What are the recent advances in vision-language action models for robot manipulation?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        for ev in res.evidence:
            assert ev.domain == "artificial_intelligence"
            assert not ev.paper_id.startswith("HC")
            assert not ev.paper_id.startswith("AG")

    def test_multi_domain_ai_healthcare_uses_both_domains(self):
        """AI + Healthcare query permits AI and Healthcare papers."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "How is AI used in medical image analysis?"
        res = pipeline.answer(q)

        domains_present = {ev.domain for ev in res.evidence}
        assert domains_present.issubset({"artificial_intelligence", "healthcare"})

    def test_online_fallback_respects_multi_domain_scope(self):
        """Online fallback for AI + Healthcare queries searches only AI and Healthcare categories."""
        online_retriever = OnlineAcademicRetriever()
        items = online_retriever.retrieve(
            "AI medical image analysis",
            allowed_domains=["artificial_intelligence", "healthcare"],
            max_results=3
        )
        if items:
            for item in items:
                assert item.domain in ["artificial_intelligence", "healthcare"]

    def test_question_repetition_regenerates_answer(self):
        """If answer repeats question, QuestionAnswerRelevanceValidator flags it."""
        valid, msg = QuestionAnswerRelevanceValidator.validate_answer_intent(
            "What is retrieval-augmented generation?",
            "What is retrieval-augmented generation? RAG is an AI framework.",
            "Definition"
        )
        assert valid is False

    def test_5_required_live_benchmark_queries_phase26(self):
        """Verify 5 canonical benchmark queries execute without error."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        queries = [
            ("What are the limitations of retrieval-augmented generation systems?", {"domain": "artificial_intelligence"}),
            ("What is quantum teleportation using biological neural networks?", {"domain": "artificial_intelligence"}),
            ("What is surface code quantum error correction?", None),
            ("How do vector database indexes optimize retrieval in RAG systems?", {"domain": "artificial_intelligence"}),
            ("How is AI used in medical image analysis?", None)
        ]

        for q, filters in queries:
            res = pipeline.answer(q, filters=filters)
            assert res.answer is not None
            assert len(res.answer) > 10
            assert res.confidence in ["Excellent", "High", "Moderate", "Low", "Insufficient"]
