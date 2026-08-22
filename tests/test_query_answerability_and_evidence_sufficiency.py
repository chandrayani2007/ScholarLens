"""
Regression & Unit Test Suite for Query Answerability and Evidence Sufficiency

Tests:
1. Correct local answer (in-corpus with direct supporting passages).
2. Related-but-not-answerable local evidence -> triggers online fallback.
3. Partial local evidence -> triggers online fallback to supplement evidence.
4. Wrong-domain evidence -> triggers online fallback.
5. Single-domain insufficient evidence -> triggers online fallback.
6. Multi-domain insufficient evidence -> triggers online fallback.
7. Completely out-of-corpus query -> pure online fallback.
8. Citation-to-claim alignment.
9. No unsupported claims.
10. Why This Answer counts only actually supporting sources.
11. No hardcoded topic triggers (generic decomposition and answerability).
12. Online fallback works for previously unseen questions.
"""

import pytest
from src.pipeline.rag import (
    RAGPipeline,
    EvidenceSufficiencyEvaluator,
    RelevanceGate,
    DomainScopeDetector,
    FinalSafetyGate,
)
from src.pipeline.retrieval import RetrievalResult
from src.pipeline.online_fallback import OnlineAcademicRetriever, OnlineEvidenceItem


@pytest.fixture
def rag_pipeline():
    return RAGPipeline()


class TestQueryAnswerabilityAndEvidenceSufficiency:
    """Test suite ensuring that query answerability (not mere topic relatedness) controls online fallback."""

    def test_1_correct_local_answer_for_sufficient_in_corpus_query(self, rag_pipeline):
        """In-corpus queries with direct supporting local passages use local corpus evidence."""
        res = rag_pipeline.answer("How do IoT sensors optimize smart irrigation?")
        assert res.answer is not None
        assert "insufficient evidence" not in res.answer.lower()
        assert len(res.citations) >= 1
        # Citations should be local [E#]
        assert any(k.startswith("E") for k in res.citations.keys())
        assert res.why_this_answer.evidence_strength in ("Excellent", "High", "Moderate")

    def test_2_related_but_unanswerable_triggers_online_fallback(self, rag_pipeline):
        """
        Critical regression test:
        'How does Explainable AI improve trust in healthcare?'
        Local corpus contains general XAI papers (AI036/AI037), but lacks direct healthcare trust mechanism.
        System MUST mark local evidence as insufficient and activate Online Academic Search.
        """
        res = rag_pipeline.answer("How does Explainable AI improve trust in healthcare?")
        assert res.answer is not None
        assert res.retrieval_metadata.get("online_fallback_active") is True
        # Must not cite unrelated AI papers
        assert not any("AI037" in str(p) for p in res.why_this_answer.contributing_papers)

    def test_3_partial_local_evidence_triggers_online_fallback(self, rag_pipeline):
        """When local evidence only partially answers a multi-aspect query, online fallback activates."""
        res = rag_pipeline.answer("What are the security risks of federated learning in healthcare?")
        assert res.answer is not None
        assert res.retrieval_metadata.get("online_fallback_active") is True

    def test_4_wrong_domain_evidence_triggers_online_fallback(self, rag_pipeline):
        """If retrieved passages belong to an unrelated domain, answerability evaluator rejects them."""
        # Create mock results from wrong domain (agriculture for a quantum computing query)
        fake_results = [
            RetrievalResult(
                unit_id="AG001_U01",
                parent_chunk_id="AG001_C01",
                chunk_id="AG001_C01",
                paper_id="AG001",
                section_id="SEC_01",
                section_name="Introduction",
                domain="agriculture",
                subtopic="crop",
                page_start=1,
                page_end=1,
                text="Crop yield optimization using precision irrigation sensors and nitrogen fertilizers.",
                dense_score=0.2,
                bm25_score=1.0,
                rrf_score=0.01,
                rank=1,
                token_count=50,
                retrieval_methods=["bm25"]
            )
        ]
        is_rel, is_ans, rationale, score = EvidenceSufficiencyEvaluator.evaluate(
            "What is quantum teleportation?", fake_results
        )
        assert is_rel is False or is_ans is False
        assert score < 0.50

    def test_5_single_domain_insufficient_evidence_triggers_online_fallback(self, rag_pipeline):
        """Single-domain questions with inadequate local evidence trigger online fallback."""
        res = rag_pipeline.answer("What are recent advances in tokamak fusion energy confinement?")
        assert res.answer is not None
        assert res.retrieval_metadata.get("online_fallback_active") is True

    def test_6_multi_domain_insufficient_evidence_triggers_online_fallback(self, rag_pipeline):
        """Cross-domain query without direct local intersection triggers online fallback."""
        res = rag_pipeline.answer("How do vision-language models assist robotic surgery in healthcare?")
        assert res.answer is not None
        assert res.retrieval_metadata.get("online_fallback_active") is True

    def test_7_completely_out_of_corpus_query_uses_pure_online_fallback(self, rag_pipeline):
        """Completely out-of-corpus queries use pure online academic search."""
        res = rag_pipeline.answer("What is quantum teleportation?")
        assert res.answer is not None
        assert res.retrieval_metadata.get("online_fallback_active") is True
        local_cits = [k for k in res.citations.keys() if k.startswith("E")]
        assert len(local_cits) == 0

    def test_8_citation_to_claim_alignment(self, rag_pipeline):
        """Every citation tag in the answer text must map to an active entry in the citations dictionary."""
        res = rag_pipeline.answer("How does Explainable AI improve trust in healthcare?")
        import re
        tags_in_text = re.findall(r"\[([EO]\d+)\]", res.answer)
        for tag in tags_in_text:
            assert tag in res.citations

    def test_9_no_unsupported_claims(self, rag_pipeline):
        """Answers must have 0 unsupported claims reported in WhyThisAnswer."""
        res = rag_pipeline.answer("How does Explainable AI improve trust in healthcare?")
        assert res.why_this_answer.unsupported_claims == 0

    def test_10_why_this_answer_counts_only_supporting_sources(self, rag_pipeline):
        """Why This Answer bullet points and metadata count only cited contributing sources."""
        res = rag_pipeline.answer("How does Explainable AI improve trust in healthcare?")
        local_count = sum(1 for k in res.citations.keys() if k.startswith("E"))
        online_count = sum(1 for k in res.citations.keys() if k.startswith("O"))

        # Verify bullet points reflect exact counts
        bullets_text = " ".join(res.why_this_answer.bullet_points)
        assert f"Local evidence: {local_count} passage(s)" in bullets_text
        assert f"Online evidence: {online_count} academic source(s)" in bullets_text

    def test_11_no_hardcoded_topic_triggers(self):
        """Generic query decomposer parses arbitrary unseen queries correctly."""
        unseen_q = "How does neuromorphic spike timing improve underwater acoustic localization?"
        decomp = EvidenceSufficiencyEvaluator.decompose_query(unseen_q)
        assert len(decomp["subject_terms"]) >= 1
        assert "timing" in decomp["all_content_words"] or "neuromorphic" in decomp["all_content_words"]

    def test_12_unseen_query_triggers_online_fallback(self, rag_pipeline):
        """Unseen queries not in the local corpus trigger online academic fallback and return real papers."""
        res = rag_pipeline.answer("How does reinforcement learning improve robotic manipulation?")
        assert res.answer is not None
        assert res.retrieval_metadata.get("online_fallback_active") is True

    def test_13_related_but_unanswerable_algorithm_query_rejected(self):
        """
        Generic answerability test:
        Question: 'What algorithms are commonly used for intrusion detection?'
        Passage: 'Machine learning is widely used for intrusion detection.'
        Expected: Related=True, Answerable=False (lacks algorithm information).
        """
        fake_passage = RetrievalResult(
            unit_id="CY001_U01",
            parent_chunk_id="CY001_C01",
            chunk_id="CY001_C01",
            paper_id="CY001",
            section_id="SEC_01",
            section_name="Introduction",
            domain="cybersecurity",
            subtopic="network",
            page_start=1,
            page_end=1,
            text="Machine learning is widely used for intrusion detection in enterprise network security systems.",
            dense_score=0.85,
            bm25_score=10.0,
            rrf_score=0.03,
            rank=1,
            token_count=50,
            retrieval_methods=["bm25", "dense"]
        )
        res = EvidenceSufficiencyEvaluator.evaluate_detailed(
            "What algorithms are commonly used for intrusion detection?",
            [fake_passage]
        )
        assert res.is_related is True
        assert res.is_answerable is False
        assert res.decision in ("NOT_ANSWERABLE", "PARTIALLY_ANSWERABLE", "LOCAL_INSUFFICIENT")

    def test_14_missing_aspect_limitation_query_rejected(self):
        """
        Generic answerability test:
        Question: 'What are the limitations of retrieval-augmented generation?'
        Passage only discusses benefits of RAG without mentioning limitations or bottlenecks.
        Expected: Answerable=False.
        """
        fake_passage = RetrievalResult(
            unit_id="AI001_U01",
            parent_chunk_id="AI001_C01",
            chunk_id="AI001_C01",
            paper_id="AI001",
            section_id="SEC_01",
            section_name="Introduction",
            domain="artificial_intelligence",
            subtopic="rag",
            page_start=1,
            page_end=1,
            text="Retrieval-augmented generation enhances output quality and provides superior factual accuracy.",
            dense_score=0.90,
            bm25_score=12.0,
            rrf_score=0.04,
            rank=1,
            token_count=50,
            retrieval_methods=["bm25", "dense"]
        )
        res = EvidenceSufficiencyEvaluator.evaluate_detailed(
            "What are the limitations of retrieval-augmented generation?",
            [fake_passage]
        )
        assert res.is_related is True
        assert res.is_answerable is False
        assert res.decision in ("NOT_ANSWERABLE", "PARTIALLY_ANSWERABLE", "LOCAL_INSUFFICIENT")
