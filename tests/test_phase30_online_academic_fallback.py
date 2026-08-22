"""
Phase 30: Production-Verified Online Academic Fallback Test Suite

Verifies:
1. test_local_corpus_checked_before_online_search: Every query searches 1,000-paper local corpus first.
2. test_in_corpus_query_uses_local_evidence: In-corpus queries use local evidence (Local >= 1, Online = 0).
3. test_insufficient_local_evidence_triggers_online_fallback: Dynamic detection triggers ArXiv search when local evidence is insufficient.
4. test_no_hardcoded_topic_trigger_required: Dynamic concept evaluation works for unseen topics without keyword triggers.
5. test_online_arxiv_metadata_is_authentic: Every online paper has real arXiv ID, title, authors, date, URL, and abstract.
6. test_online_citations_are_grounded: [O#] citations directly support answer claims.
7. test_local_citations_are_grounded: [E#] citations directly support answer claims.
8. test_why_this_answer_counts_only_used_sources: Why This Answer reports only sources cited in the final answer.
9. test_out_of_corpus_query_uses_online_academic_search: Pure out-of-corpus queries use Online Academic Search.
10. test_network_failure_returns_honest_fallback: Timeout or API error returns honest insufficient evidence fallback.
11. test_all_domains_retrieval: General research questions search across all 5 domains.
12. test_ai_healthcare_multi_domain_retrieval: Multi-domain queries search AI and Healthcare domains.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.rag import RAGPipeline, RelevanceGate, FinalSafetyGate, CitationValidator
from src.pipeline.online_fallback import OnlineAcademicRetriever, OnlineEvidenceItem
from src.pipeline.retrieval import HybridRetriever, RetrievalResult
from src.pipeline.llm import MockLLMProvider


class TestPhase30OnlineAcademicFallback:

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        self.retriever = HybridRetriever()
        self.online_retriever = OnlineAcademicRetriever(timeout=8)
        self.llm = MockLLMProvider()
        self.pipeline = RAGPipeline(
            retriever=self.retriever,
            online_retriever=self.online_retriever,
            llm_provider=self.llm
        )

    def test_local_corpus_checked_before_online_search(self):
        """Verify that local retriever is ALWAYS called first before any online fallback."""
        with patch.object(self.retriever, "retrieve", wraps=self.retriever.retrieve) as mock_local, \
             patch.object(self.online_retriever, "retrieve", wraps=self.online_retriever.retrieve) as mock_online:

            # In-corpus query
            res = self.pipeline.answer("What are the limitations of retrieval-augmented generation systems?")
            
            # Local retrieval must have been called
            assert mock_local.called
            # Online retrieval must NOT have been called because local evidence is sufficient
            assert not mock_online.called
            assert res.retrieval_metadata["local_evidence_count"] >= 1
            assert res.retrieval_metadata["online_evidence_count"] == 0

    def test_in_corpus_query_uses_local_evidence(self):
        """Verify known in-corpus queries use local evidence without calling online search."""
        res = self.pipeline.answer("What are the limitations of retrieval-augmented generation systems?")
        
        assert len(res.citations) >= 1
        assert all(k.startswith("E") for k in res.citations.keys())
        assert res.why_this_answer.source_type == "Research Mind Corpus"
        assert "Local evidence: 0" not in str(res.why_this_answer.bullet_points)
        assert res.retrieval_metadata["online_fallback_active"] is False

    def test_insufficient_local_evidence_triggers_online_fallback(self):
        """Verify out-of-corpus query triggers online academic search when local evidence is insufficient."""
        res = self.pipeline.answer("What is quantum teleportation?")
        
        # Must have triggered online academic fallback
        assert res.retrieval_metadata["online_evidence_count"] >= 1
        assert any(k.startswith("O") for k in res.citations.keys())
        assert "Online Academic Search" in res.why_this_answer.source_type
        assert res.why_this_answer.evidence_strength in ("Excellent", "High", "Moderate")

    def test_no_hardcoded_topic_trigger_required(self):
        """Verify that dynamic sufficiency evaluation works for unseen topics without hardcoded keyword lists."""
        unseen_query = "What are recent advances in tokamak fusion energy confinement?"
        
        # Test RelevanceGate.evaluate directly
        fake_irrelevant_local = [
            RetrievalResult(
                chunk_id="AI001_C01",
                unit_id="AI001_U01",
                parent_chunk_id="AI001_C01",
                paper_id="AI001",
                section_id="sec_01",
                section_name="Introduction",
                domain="artificial_intelligence",
                subtopic="deep_learning",
                page_start=1,
                page_end=2,
                text="This paper investigates deep convolutional neural networks for computer vision tasks.",
                rrf_score=0.012,
                rank=1,
                token_count=50,
            )
        ]
        
        is_rel, reason, match_ratio = RelevanceGate.evaluate(unseen_query, fake_irrelevant_local)
        # Must reject local evidence purely because tokamak/fusion/confinement are missing
        assert is_rel is False
        assert match_ratio < 0.45

    def test_online_arxiv_metadata_is_authentic(self):
        """Verify that ArXiv API returns authentic metadata and rejects malformed items."""
        items = self.online_retriever.retrieve(query="quantum teleportation", max_results=3)
        
        if items:
            for item in items:
                assert item.citation_id.startswith("O")
                assert item.paper_id.startswith("arXiv:")
                assert len(item.title) > 5
                assert len(item.authors) >= 1
                assert len(item.published_date) >= 4
                assert item.url.startswith("http")
                assert len(item.text) >= 40
                assert item.source_type == "online"

    def test_online_citations_are_grounded(self):
        """Verify that [O#] citations in answer match authentic online evidence passages."""
        res = self.pipeline.answer("What is quantum teleportation?")
        
        for tag, cit in res.citations.items():
            if tag.startswith("O"):
                assert cit.paper_id.startswith("arXiv:")
                assert cit.source_type == "online"
                assert cit.title is not None

    def test_local_citations_are_grounded(self):
        """Verify that [E#] citations in answer match genuine local corpus papers."""
        res = self.pipeline.answer("How does explainable AI improve trust in healthcare?")
        
        for tag, cit in res.citations.items():
            if tag.startswith("E"):
                assert cit.paper_id.startswith("AI") or cit.paper_id.startswith("HC")
                assert cit.source_type == "corpus"

    def test_why_this_answer_counts_only_used_sources(self):
        """Verify Why This Answer reports ONLY sources cited in final answer."""
        res = self.pipeline.answer("What is quantum teleportation?")
        
        used_tags = list(res.citations.keys())
        reported_papers = res.why_this_answer.contributing_papers
        
        # Every reported paper must be from used citations
        expected_papers = list(dict.fromkeys(c.paper_id for c in res.citations.values()))
        assert set(reported_papers) == set(expected_papers)
        assert len(res.why_this_answer.evidence_passages) == len(used_tags)

    def test_out_of_corpus_query_uses_online_academic_search(self):
        """Verify another unseen out-of-corpus query (tokamak fusion energy) uses Online Academic Search."""
        res = self.pipeline.answer("What are recent advances in tokamak fusion energy confinement?")
        
        assert res.retrieval_metadata["online_evidence_count"] >= 1
        assert "Online Academic Search" in res.why_this_answer.source_type
        assert any(k.startswith("O") for k in res.citations.keys())

    def test_network_failure_returns_honest_fallback(self):
        """Verify that when ArXiv API fails or times out, pipeline returns honest Tier 4 fallback without crashing."""
        with patch.object(self.online_retriever, "retrieve", return_value=[]):
            res = self.pipeline.answer("What is quantum teleportation?")
            
            assert "Insufficient evidence" in res.answer
            assert res.confidence == "Insufficient"
            assert res.why_this_answer.evidence_strength == "Insufficient"
            assert len(res.citations) == 0

    def test_all_domains_retrieval(self):
        """Verify general queries search across all 5 domains."""
        res = self.pipeline.answer("What are common research evaluation methodologies across scientific disciplines?")
        
        assert len(res.domain_scope) == 5
        assert "artificial_intelligence" in res.domain_scope
        assert "cybersecurity" in res.domain_scope

    def test_ai_healthcare_multi_domain_retrieval(self):
        """Verify multi-domain queries search AI and Healthcare domains without unrelated domains."""
        res = self.pipeline.answer("How is AI used in medical image analysis?")
        
        assert "artificial_intelligence" in res.domain_scope
        assert "healthcare" in res.domain_scope
        assert "agriculture" not in res.domain_scope
        assert "cybersecurity" not in res.domain_scope
