"""
Phase 19/24 — Semantic Retrieval, Claim-Level Grounding & Answer Quality Test Suite
"""

import pytest
from src.pipeline.rag import RAGPipeline, RelevanceGate, ClaimGroundingValidator
from src.pipeline.llm import MockLLMProvider


class TestPhase19AnswerQualityAndGrounding:
    def test_photosynthesis_semantic_grounded_answer(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is photosynthesis?")

        assert res.answer
        assert res.answer.startswith("Photosynthesis is")
        assert "[E1]" in res.answer or "[E2]" in res.answer
        assert len(res.citations) > 0
        assert "insufficient evidence" not in res.answer.lower()

    def test_healthcare_semantic_retrieval_and_answer(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How is deep learning used in medical image diagnosis?", filters={"domain": "healthcare"})

        assert res.answer
        assert res.answer.startswith("Deep learning is used in medical image diagnosis") or res.answer.startswith("Convolutional neural networks")
        assert "[E1]" in res.answer or "[E2]" in res.answer
        assert len(res.citations) > 0
        assert res.evidence[0].domain == "healthcare"

    def test_claim_grounding_validator(self):
        evidence_map = {
            "E1": type("EvidenceItem", (), {"text": "Deep learning models evaluate high-dimensional network telemetry and system logs to identify intrusion anomalies."})()
        }
        answer = "Deep learning detects cyber attacks by analyzing high-dimensional network telemetry [E1]. It also enables quantum teleportation of interstellar data."
        
        grounded, unsupported_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(answer, evidence_map)
        assert "telemetry" in grounded
        assert "E1" in active_tags

    def test_negative_out_of_corpus_query(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is neuro-synaptic warp propulsion in deep space exploration?")

        assert "Insufficient evidence was found" in res.answer
        assert res.confidence in ["Low", "Insufficient"]
        assert res.why_this_answer.evidence_strength in ["Weak", "Insufficient"]
        assert len(res.citations) == 0

    def test_7_required_canonical_queries(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())

        queries = [
            ("What is retrieval augmented generation?", ("Retrieval-Augmented Generation (RAG)", "Retrieval-Augmented Generation")),
            ("How can IoT improve smart irrigation?", ("IoT", "Smart")),
            ("How can deep learning detect cyber attacks?", ("Deep learning detects cyber attacks", "Deep learning")),
            ("How is deep learning used in medical image diagnosis?", ("Deep learning is used in medical image diagnosis", "Convolutional neural networks")),
            ("How can machine learning help predict climate patterns?", ("Climate modeling uses machine learning", "Machine learning models")),
            ("What is photosynthesis?", ("Photosynthesis is", "Photosynthesis")),
        ]

        for q, expected_prefixes in queries:
            res = pipeline.answer(q)
            assert res.answer
            assert any(res.answer.startswith(prefix) for prefix in expected_prefixes)
            assert "[E1]" in res.answer or "[E2]" in res.answer
            assert len(res.citations) > 0
            assert "insufficient evidence" not in res.answer.lower()
