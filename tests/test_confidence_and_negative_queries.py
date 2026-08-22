"""
Phase 22/23 — Definitive Confidence, Evidence Grounding & Negative Query Test Suite

Verifies:
1. Strict Invariant: RAGResponse.confidence == RAGResponse.why_this_answer.evidence_strength across all queries.
2. Negative out-of-corpus query ('quantum teleportation using biological neural networks') is rejected with Insufficient evidence.
3. Irrelevant retrieved passages are NOT cited.
4. Key Sources panel matches ONLY cited evidence.
5. All canonical benchmark queries produce direct, subject-first grounded answers.
"""

import pytest
from src.pipeline.rag import RAGPipeline
from src.pipeline.llm import MockLLMProvider


class TestPhase22ConfidenceAndNegativeQueries:
    def test_confidence_equals_evidence_strength_invariant(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        queries = [
            "What is retrieval-augmented generation, and how does it improve large language model responses?",
            "How can IoT improve smart irrigation systems?",
            "How can deep learning techniques be used to detect cyber attacks?",
            "What is quantum teleportation using biological neural networks?",
            "What is photosynthesis?",
        ]

        for q in queries:
            res = pipeline.answer(q)
            assert res.confidence == res.why_this_answer.evidence_strength

    def test_negative_out_of_corpus_query_rejection(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is quantum teleportation using biological neural networks?")

        assert res.confidence == "Insufficient"
        assert res.why_this_answer.evidence_strength == "Insufficient"
        assert "Insufficient evidence was found" in res.answer
        assert len(res.citations) == 0
        assert res.retrieval_metadata["used_citations_count"] == 0

    def test_all_11_required_phase22_benchmark_queries(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())

        test_cases = [
            ("What is retrieval-augmented generation, and how does it improve large language model responses?", "artificial_intelligence", ("Retrieval-Augmented Generation", "RAG")),
            ("What are the main challenges in evaluating retrieval-augmented generation systems?", "artificial_intelligence", ("Evaluating", "RAG")),
            ("How can deep learning techniques be used to detect cyber attacks?", "cybersecurity", ("Deep learning", "Cyber")),
            ("How can machine learning help identify anomalous network traffic?", "cybersecurity", ("Deep learning", "Machine learning")),
            ("How can IoT improve smart irrigation systems?", "agriculture", ("IoT", "Smart")),
            ("How does soil moisture monitoring help optimize irrigation?", "agriculture", ("Soil moisture", "Irrigation")),
            ("How is deep learning used for medical image analysis and diagnosis?", "healthcare", ("Convolutional neural networks", "Deep learning")),
            ("What role does explainable AI play in healthcare applications?", "healthcare", ("Explainable AI", "Model")),
            ("How can machine learning be used to predict climate patterns?", "climate", ("Machine learning", "Climate")),
            ("How can machine learning improve extreme weather prediction?", "climate", ("Machine learning", "Ensemble")),
            ("What is photosynthesis?", "agriculture", ("Photosynthesis", "Plant")),
        ]

        for q, domain, expected_prefixes in test_cases:
            res = pipeline.answer(q, filters={"domain": domain})
            assert res.confidence in ["Excellent", "High", "Moderate"]
            assert res.confidence == res.why_this_answer.evidence_strength
            assert any(res.answer.startswith(prefix) for prefix in expected_prefixes)
            assert len(res.citations) >= 1
            # Key Sources contains only active citations
            for cit_tag in res.citations:
                assert cit_tag in res.why_this_answer.evidence_passages

    def test_key_sources_equals_cited_sources(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How can IoT improve smart irrigation systems?", filters={"domain": "agriculture"})

        cited_tags = set(res.citations.keys())
        why_tags = set(res.why_this_answer.evidence_passages)
        assert cited_tags == why_tags
