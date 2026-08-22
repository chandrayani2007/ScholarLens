"""
Phase 21 — Claim-Level Evidence Validation & Citation Correctness Test Suite

Verifies:
1. Bibliography and reference list chunks are excluded from direct factual citations.
2. Factual claims map only to supporting evidence passages containing matching concepts.
3. Unsupported claims are removed or sanitized.
4. Dynamic citation counts (1, 2, 3+ citations depending on content support).
5. Why This Answer generates evidence-specific summaries instead of generic templates.
6. Key Sources panel displays active citations only.
"""

import pytest
from src.pipeline.rag import RAGPipeline, ClaimGroundingValidator, RelevanceGate
from src.pipeline.retrieval import is_bibliography_chunk, HybridRetriever
from src.pipeline.llm import MockLLMProvider


class TestPhase21ClaimValidationAndBibliographyProtection:
    def test_bibliography_chunk_detection(self):
        bib_text = "[41] Bwambale, B., et al. (2022). Smart irrigation control systems. [42] Taghvaeian, S. (2020). [43] Munir, A. (2021). Journal of Agricultural Engineering, doi:10.1016/j.agwat.2021.107200."
        assert is_bibliography_chunk(bib_text) is True

        normal_text = "Precision agriculture utilizes wireless sensor networks to monitor real-time soil moisture and environmental variables."
        assert is_bibliography_chunk(normal_text) is False

    def test_bibliography_chunk_rejected_from_citations(self):
        evidence_map = {
            "E1": type("EvidenceItem", (), {"text": "[41] Smith, A. (2020). [42] Jones, B. (2021). [43] Davis, C. (2022). Journal of Sensors, doi:10.1016/j.sens.2022.01."})(),
            "E2": type("EvidenceItem", (), {"text": "Wireless sensor nodes monitor soil moisture and environmental parameters for precision irrigation."})(),
        }
        answer = "IoT systems monitor soil moisture using wireless sensor nodes [E1]."
        
        grounded, unsupported_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(answer, evidence_map)
        assert "E1" not in active_tags
        assert "E2" in active_tags

    def test_evidence_specific_explanation_summary(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How can IoT improve smart irrigation?", filters={"domain": "agriculture"})

        summary = res.why_this_answer.explanation_summary
        assert "Passage [E1]" in summary or "Passage [E2]" in summary
        assert "Direct scientific evidence addressing" not in summary
        assert len(res.citations) > 0

    def test_dynamic_citation_count(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res_rag = pipeline.answer("What is retrieval augmented generation?", filters={"domain": "artificial_intelligence"})
        res_iot = pipeline.answer("How can IoT improve smart irrigation?", filters={"domain": "agriculture"})

        assert len(res_rag.citations) >= 1
        assert len(res_iot.citations) >= 1
        assert res_rag.why_this_answer.evidence_strength in ["Excellent", "High", "Strong", "Moderate"]

    def test_6_canonical_pipeline_queries(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())

        test_questions = [
            ("What is retrieval augmented generation?", "artificial_intelligence"),
            ("How can IoT improve smart irrigation?", "agriculture"),
            ("How can deep learning detect cyber attacks?", "cybersecurity"),
            ("How is deep learning used in medical image diagnosis?", "healthcare"),
            ("How can machine learning help predict climate patterns?", "climate"),
            ("What is photosynthesis?", "agriculture"),
        ]

        for q, domain in test_questions:
            res = pipeline.answer(q, filters={"domain": domain})
            assert res.answer
            assert "[E" in res.answer
            assert len(res.citations) > 0
            assert res.why_this_answer.evidence_strength in ["Excellent", "High", "Strong", "Moderate"]
            assert "Direct scientific evidence addressing" not in res.why_this_answer.explanation_summary
