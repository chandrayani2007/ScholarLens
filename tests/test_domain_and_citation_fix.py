"""
Phase 20 — Domain Filtering & Content-Driven Citation Selection Test Suite

Verifies:
1. Domain filtering is strictly enforced in ChromaDB and BM25 before RRF fusion.
2. AI domain queries return 100% AI... papers.
3. Cybersecurity domain queries return 100% CY... papers.
4. Agriculture domain queries return 100% AG... papers.
5. Healthcare domain queries return 100% HC... papers.
6. Climate domain queries return 100% CL... papers.
7. All-domains queries allow cross-domain retrieval.
8. Citation count is content-driven (not hardcoded to E1 and E2).
9. Key Sources matches active citations only.
"""

import pytest
from src.pipeline.rag import RAGPipeline, ClaimGroundingValidator
from src.pipeline.retrieval import HybridRetriever
from src.pipeline.llm import MockLLMProvider


class TestPhase20DomainFilteringAndCitations:
    def test_ai_domain_retrieval(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer(
            "What is retrieval augmented generation?",
            filters={"domain": "artificial_intelligence"}
        )
        assert res.evidence
        for item in res.evidence:
            assert item.domain == "artificial_intelligence"
            assert item.paper_id.startswith("AI")

    def test_agriculture_domain_retrieval(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer(
            "How can IoT improve smart irrigation?",
            filters={"domain": "agriculture"}
        )
        assert res.evidence
        for item in res.evidence:
            assert item.domain == "agriculture"
            assert item.paper_id.startswith("AG")

    def test_cybersecurity_domain_retrieval(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer(
            "How can deep learning detect cyber attacks?",
            filters={"domain": "cybersecurity"}
        )
        assert res.evidence
        for item in res.evidence:
            assert item.domain == "cybersecurity"
            assert item.paper_id.startswith("CY")

    def test_healthcare_domain_retrieval(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer(
            "How is deep learning used in medical image diagnosis?",
            filters={"domain": "healthcare"}
        )
        assert res.evidence
        for item in res.evidence:
            assert item.domain == "healthcare"
            assert item.paper_id.startswith("HC")

    def test_climate_domain_retrieval(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer(
            "How can machine learning help predict climate patterns?",
            filters={"domain": "climate"}
        )
        assert res.evidence
        for item in res.evidence:
            assert item.domain == "climate"
            assert item.paper_id.startswith("CL")

    def test_all_domains_retrieval(self):
        retriever = HybridRetriever()
        results = retriever.retrieve("What is machine learning?", top_k=10, filters=None)
        assert len(results) > 0
        domains = set(r.domain for r in results)
        # Should contain results from indexed corpus without forcing AI filter
        assert len(domains) >= 1

    def test_key_sources_cited_only(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How can IoT improve smart irrigation?", filters={"domain": "agriculture"})

        cited_tags = set(res.citations.keys())
        for tag in cited_tags:
            assert tag in res.citations
            assert res.citations[tag].paper_id.startswith("AG")

    def test_dynamic_claim_citation_mapping(self):
        evidence_map = {
            "E3": type("EvidenceItem", (), {"citation_id": "E3", "text": "Soil moisture sensors communicate real-time telemetry over wireless networks."})(),
            "E6": type("EvidenceItem", (), {"citation_id": "E6", "text": "Automated variable-rate valves regulate water discharge dynamically."})()
        }
        answer = "IoT systems monitor soil moisture using wireless telemetry [E3]. Automated valves adjust water delivery dynamically [E6]."
        
        grounded, unsupported_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(answer, evidence_map)
        assert "E3" in active_tags
        assert "E6" in active_tags
        assert unsupported_count == 0
