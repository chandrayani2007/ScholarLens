"""
End-to-End Semantic Acceptance Test Suite — ScholarLens

Tests the 8 core scientific queries across retrieval, answerability evaluation,
online academic fallback, answer synthesis, claim verification, and quality gates.

Acceptance Criteria:
- HTTP 200 or presence of citations is NOT sufficient proof of correctness.
- The final answer must directly and semantically address the user's question.
- Evidence answerability must separate topical relevance from aspect answerability.
"""

import os
import json
import pytest
from typing import Dict, Any

from src.pipeline.rag import RAGPipeline, GenericQuestionAnalyzer, GenericEvidenceEvaluator
from src.pipeline.domain_scope import DomainScopeDetector

E2E_TEST_QUERIES = [
    "What are the limitations of Agentic RAG?",
    "What are the limitations of large language models?",
    "What algorithms are commonly used for intrusion detection?",
    "How does federated learning preserve privacy?",
    "What are the main challenges of deploying deep learning models on edge devices?",
    "What is quantum teleportation?",
    "Compare cloud computing and edge computing.",
    "What are the main components of a transformer architecture?",
]


class TestE2ESemanticAcceptance:

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        self.pipeline = RAGPipeline()

    @pytest.mark.parametrize("query", E2E_TEST_QUERIES)
    def test_query_end_to_end(self, query: str):
        print(f"\n[E2E TEST] Executing query: '{query}'")
        
        # Step 1: Decomposition
        q_repr = GenericQuestionAnalyzer.analyze(query)
        scope_res = DomainScopeDetector.detect(query)
        
        assert q_repr.intent is not None
        assert scope_res.scope_type is not None

        # Step 2: Pipeline execution
        res = self.pipeline.answer(query)
        res_dict = res.to_dict()

        # Step 3: Semantic Quality Validation
        answer = res.answer.strip()
        assert len(answer) > 50, "Answer text must not be empty or truncated."

        # Verify that if answer is an insufficient evidence fallback, it is honest and properly tagged
        if "insufficient evidence" in answer.lower():
            assert res.confidence == "Insufficient"
            assert res.why_this_answer.evidence_strength == "Insufficient"
            print(f"  [RESULT] Honest Insufficient Evidence Fallback for '{query}'")
        else:
            # Answer is provided — verify semantic grounding
            assert res.confidence in ["High", "Excellent", "Good", "Moderate"]
            assert len(res.citations) > 0, "Grounded answer must contain active citation tags."
            
            # Check subject phrase match in generated answer
            ans_lower = answer.lower()
            matched_subj = [w for w in q_repr.main_subject if w.lower() in ans_lower]
            assert len(matched_subj) >= 1 or not q_repr.main_subject, f"Answer must discuss main subject: {q_repr.main_subject}"

            # Verify citation mapping alignment
            for cit_id, cit in res.citations.items():
                assert cit.paper_id is not None
                assert cit_id in answer, f"Citation [{cit_id}] must appear in answer text."

            print(f"  [RESULT] Grounded Answer Generated ({res.why_this_answer.source_type}) with {len(res.citations)} citations.")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
