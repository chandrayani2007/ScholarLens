"""
Phase 23 — Final Question-Repetition & Answer-Intent Validation Test Suite

Verifies:
1. Answers do NOT repeat, paraphrase, or start with the user's question.
2. Answers directly address the question intent across 15 fine-grained intents.
3. Generic template boilerplate is rejected.
4. Question-Answer semantic relevance is enforced (wrong-question answers receive Insufficient confidence).
5. Claim-level validation verifies both claim grounding and question relevance.
6. Dynamic citations and Key Sources strictly match active citations.
7. Comprehensive testing of all 15 benchmark questions + negative query.
"""

import pytest
from src.pipeline.rag import RAGPipeline, QuestionAnswerRelevanceValidator, RelevanceGate, ClaimGroundingValidator
from src.pipeline.llm import MockLLMProvider


class TestPhase23AnswerIntentValidation:
    def test_answer_does_not_repeat_question(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What are the limitations of evaluating RAG systems?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        assert not res.answer.lower().startswith(q.lower().rstrip("?"))
        assert "what are the limitations of evaluating rag systems" not in res.answer.lower()

    def test_answer_directly_addresses_question(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "How can IoT sensors help farmers control irrigation?"
        res = pipeline.answer(q, filters={"domain": "agriculture"})

        assert res.answer.startswith("IoT sensors help farmers control irrigation by")
        assert len(res.citations) >= 1

    def test_answer_intent_matches_question(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        
        test_intents = [
            ("What is retrieval-augmented generation?", "Definition"),
            ("How can deep learning identify malicious network traffic?", "Detection"),
            ("Why does RAG reduce hallucinations?", "Cause"),
            ("How does explainable AI improve trust in healthcare?", "Effect"),
            ("What are the main advantages of smart irrigation?", "Advantage"),
            ("What are the limitations of evaluating RAG systems?", "Limitation"),
            ("How are CNNs used in medical image analysis?", "Application"),
            ("What are the major challenges of climate modeling?", "Challenge"),
        ]

        for q, expected_intent in test_intents:
            detected_intent = RelevanceGate.classify_intent(q)
            assert detected_intent == expected_intent

    def test_generic_template_is_rejected(self):
        question = "How can machine learning improve extreme-weather forecasting?"
        bad_answer = "How can machine learning improve extreme-weather forecasting involves applying evidence-grounded domain models and analytical processes [E1]."
        
        is_valid, reason = QuestionAnswerRelevanceValidator.validate_answer_intent(question, bad_answer, "Prediction")
        assert is_valid is False
        assert "forbidden generic template" in reason or "exact question text" in reason or "question clause" in reason

    def test_irrelevant_evidence_is_rejected(self):
        question = "How does explainable AI improve trust in healthcare?"
        bad_answer = "Automated feature representation improves diagnostic consistency, assisting clinicians with fast evaluations [E1]."
        
        is_valid, reason = QuestionAnswerRelevanceValidator.validate_answer_intent(question, bad_answer, "Effect")
        assert is_valid is False
        assert "explainability" in reason.lower() or "trust" in reason.lower()

    def test_claim_answers_question(self):
        evidence_map = {
            "E1": type("EvidenceItem", (), {"text": "Explainable AI provides transparent feature attributions that build clinician trust in decision support systems."})()
        }
        answer = "Explainable AI builds clinician trust by providing transparent feature attributions [E1]."
        question = "How does explainable AI improve trust in healthcare?"

        grounded, unsupported, active = ClaimGroundingValidator.validate_and_filter_claims(answer, evidence_map, question)
        assert len(active) == 1
        assert "trust" in grounded.lower()

    def test_claim_is_supported_by_citation(self):
        evidence_map = {
            "E1": type("EvidenceItem", (), {"text": "Soil moisture sensors continuously measure volumetric water content for telemetry controllers."})()
        }
        answer = "Soil moisture sensors measure water content [E1]. Interstellar warp drives travel across deep space [E2]."
        question = "How do soil moisture sensors work?"

        grounded, unsupported, active = ClaimGroundingValidator.validate_and_filter_claims(answer, evidence_map, question)
        assert "warp" not in grounded.lower()

    def test_confidence_requires_question_relevance(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is quantum teleportation using biological neural networks?")
        
        assert res.confidence == "Insufficient"
        assert res.why_this_answer.evidence_strength == "Insufficient"

    def test_key_sources_match_citations(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How are CNNs used in medical image analysis?", filters={"domain": "healthcare"})

        assert set(res.citations.keys()) == set(res.why_this_answer.evidence_passages)

    def test_question_repetition_regression(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        questions = [
            "What are the limitations of evaluating RAG systems?",
            "How can deep learning identify malicious network traffic?",
            "How does explainable AI improve trust in healthcare?",
        ]

        for q in questions:
            res = pipeline.answer(q)
            assert not res.answer.lower().startswith(q.lower().rstrip("?"))
            assert "involves applying evidence-grounded domain models" not in res.answer.lower()

    def test_explainable_ai_healthcare_query(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How does explainable AI improve trust in healthcare?", filters={"domain": "healthcare"})

        assert "trust" in res.answer.lower() or "explainable" in res.answer.lower()
        assert res.confidence in ["Excellent", "High", "Moderate"]
        assert len(res.citations) >= 1

    def test_rag_limitations_query(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What are the limitations of evaluating RAG systems?", filters={"domain": "artificial_intelligence"})

        assert "evaluating" in res.answer.lower() or "difficult" in res.answer.lower() or "pipeline" in res.answer.lower()
        assert res.confidence in ["Excellent", "High", "Moderate"]

    def test_cyber_attack_detection_query(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How can deep learning identify malicious network traffic?", filters={"domain": "cybersecurity"})

        assert "malicious" in res.answer.lower() or "traffic" in res.answer.lower() or "network" in res.answer.lower()
        assert res.confidence in ["Excellent", "High", "Moderate"]

    def test_iot_irrigation_query(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("How can IoT sensors help farmers control irrigation?", filters={"domain": "agriculture"})

        assert "irrigation" in res.answer.lower() or "sensors" in res.answer.lower()
        assert res.confidence in ["Excellent", "High", "Moderate"]

    def test_photosynthesis_query(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is photosynthesis?", filters={"domain": "agriculture"})

        assert res.answer.startswith("Photosynthesis is")
        assert res.confidence in ["Excellent", "High", "Moderate"]

    def test_negative_quantum_query(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        res = pipeline.answer("What is quantum teleportation using biological neural networks?")

        assert res.confidence == "Insufficient"
        assert len(res.citations) == 0
        assert "Insufficient evidence was found" in res.answer

    def test_all_15_canonical_benchmark_questions(self):
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())

        questions = [
            ("What is retrieval-augmented generation?", "artificial_intelligence"),
            ("What are the limitations of evaluating RAG systems?", "artificial_intelligence"),
            ("How can deep learning identify malicious network traffic?", "cybersecurity"),
            ("How can IoT sensors help farmers control irrigation?", "agriculture"),
            ("How are CNNs used in medical image analysis?", "healthcare"),
            ("How can machine learning improve extreme-weather forecasting?", "climate"),
            ("What is photosynthesis?", "agriculture"),
            ("What are the major challenges of climate modeling?", "climate"),
            ("How does explainable AI improve trust in healthcare?", "healthcare"),
            ("What is quantum teleportation using biological neural networks?", None),
            ("Why does RAG reduce hallucinations?", "artificial_intelligence"),
            ("What are the main advantages of smart irrigation?", "agriculture"),
            ("What are the limitations of deep learning for cyber attack detection?", "cybersecurity"),
            ("How does climate modeling use machine learning?", "climate"),
            ("What is the role of explainability in clinical AI?", "healthcare"),
        ]

        for q, domain in questions:
            filters = {"domain": domain} if domain else None
            res = pipeline.answer(q, filters=filters)
            
            assert res.answer
            assert res.confidence == res.why_this_answer.evidence_strength
            assert set(res.citations.keys()) == set(res.why_this_answer.evidence_passages)

            if domain is None:
                assert res.confidence == "Insufficient"
                assert len(res.citations) == 0
            else:
                assert res.confidence in ["Excellent", "High", "Moderate"]
                assert len(res.citations) >= 1
                # Must NOT repeat the question or start with generic template
                assert not res.answer.lower().startswith(q.lower().rstrip("?"))
                assert "involves applying evidence-grounded domain models" not in res.answer.lower()
