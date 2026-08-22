"""
Phase 15 — Answer Generation Quality & Format Regression Suite

Verifies:
1. Answer Card contains direct, research-grounded prose answering the user's question.
2. Answer Card does NOT meta-narrate paper IDs like 'Based on paper AG021...'.
3. Inline citation tags [E1], [E2] are attached to factual claims.
4. Citations map cleanly to evidence passages.
5. Key Sources and Evidence remain separate from the main answer text.
"""

import os
import pytest
from src.pipeline.rag import RAGPipeline
from src.pipeline.llm import MockLLMProvider


class TestAnswerGenerationQuality:
    def test_direct_prose_answer_format_without_paper_id_prefix(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How can IoT improve smart irrigation?", filters={"domain": "agriculture"})

        # 1. Direct answer must exist
        assert res.answer
        assert len(res.answer) > 30

        # 2. Main answer must NOT start with 'Based on scientific evidence retrieved from paper(s)...'
        assert not res.answer.startswith("Based on scientific evidence retrieved from paper(s)")
        assert not res.answer.startswith("Based on paper")

        # 3. Main answer must contain inline bracket citations [E1] or [E2]
        assert "[E1]" in res.answer or "[E2]" in res.answer

        # 4. Citations map must contain entries for referenced tags
        assert len(res.citations) > 0
        assert "E1" in res.citations

        # 5. Evidence list must contain detailed items
        assert len(res.evidence) > 0
        assert res.evidence[0].paper_id.startswith("AG")

        # 6. WhyThisAnswer must have contributing papers separated
        assert len(res.why_this_answer.contributing_papers) > 0

    def test_cybersecurity_question_prose_format(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How can deep learning detect cyber attacks?", filters={"domain": "cybersecurity"})

        assert res.answer
        assert "cyber" in res.question.lower()
        assert not res.answer.startswith("Based on paper")
        assert "[E1]" in res.answer or "[E2]" in res.answer
        assert res.evidence[0].domain == "cybersecurity"
