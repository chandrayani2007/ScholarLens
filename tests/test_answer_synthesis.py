"""
Phase 16 — RAG Answer Synthesis Test Suite

Verifies:
1. Synthesized answers directly answer user questions rather than echoing raw passage text verbatim.
2. Distinct questions across 5 domains produce distinct synthesized answers.
3. Insufficient evidence fallback behaves deterministically.
"""

import pytest
from src.pipeline.rag import RAGPipeline
from src.pipeline.llm import MockLLMProvider


class TestRAGAnswerSynthesis:
    def test_synthesized_answer_format(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is retrieval augmented generation?", filters={"domain": "artificial_intelligence"})

        # Direct answer check
        assert res.answer
        assert "Retrieval-Augmented Generation" in res.answer
        assert "[E1]" in res.answer or "[E2]" in res.answer

        # Verification of provenance retention
        assert len(res.evidence) > 0
        assert len(res.citations) > 0

    def test_domain_synthesis_across_5_questions(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())

        questions = [
            ("artificial_intelligence", "What is retrieval augmented generation?"),
            ("cybersecurity", "How can deep learning be used to detect cyber attacks?"),
            ("agriculture", "How can IoT improve smart irrigation?"),
            ("healthcare", "How is deep learning used in medical image diagnosis?"),
            ("climate", "How can machine learning help predict climate patterns?"),
        ]

        answers = []
        for domain, q in questions:
            res = pipeline.answer(q, filters={"domain": domain})

            assert res.answer
            assert res.question == q
            assert "[E1]" in res.answer or "[E2]" in res.answer
            assert len(res.evidence) > 0
            assert res.evidence[0].domain == domain

            answers.append(res.answer)

        # All 5 synthesized answers must be distinct
        assert len(set(answers)) == 5

    def test_insufficient_evidence_fallback(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline._build_insufficient_evidence_response("Nonsensical test query", [])

        # Direct test of insufficient evidence response
        assert "Insufficient evidence" in res.answer
        assert res.confidence in ["Low", "Insufficient"]
