"""
Unit Test Suite for Question → Evidence Contract & Strict Answerability

Tests:
1. QuestionContract generation for various question intents.
2. Distinction between RELATED (topic match) and ANSWERABLE (intent match).
3. Rejection of related-only passages for Definition query ("What is cloud computing?").
4. Rejection of related-only passages for Limitation query ("What are the limitations of LLMs?").
5. Secondary targeted retrieval & online fallback trigger logic.
"""

import pytest
from src.pipeline.rag import (
    GenericQuestionAnalyzer,
    GenericEvidenceEvaluator,
    EvidenceItem,
    RAGPipeline,
)


def test_question_contract_generation():
    q_repr = GenericQuestionAnalyzer.analyze("What is cloud computing?")
    assert q_repr.intent == "Definition"
    assert q_repr.contract is not None
    assert "cloud computing" in q_repr.contract.concept
    assert "definition" in q_repr.contract.required_evidence_type.lower()


def test_cloud_computing_related_vs_answerable():
    """
    Test live failure case:
    Passages mentioning cloud computing in relation to edge computing MUST evaluate as:
    related = True
    answerable = False
    """
    q_repr = GenericQuestionAnalyzer.analyze("What is cloud computing?")

    # Passage about edge computing vs cloud computing (related, but no definition of cloud computing)
    related_passage = EvidenceItem(
        citation_id="E1",
        unit_id="CY023_U01",
        parent_chunk_id="CY023_C01",
        chunk_id="CY023_C01",
        paper_id="CY023",
        section_id="SEC_DISCUSSION",
        section_name="Discussion",
        domain="cybersecurity",
        subtopic="edge_cloud",
        page_start=5,
        page_end=6,
        text="Edge computing can bridge the gap between the deployment of cloud computing with powerful resources but high latency and onboard computing with limited computational power.",
    )

    eval_res = GenericEvidenceEvaluator.evaluate("What is cloud computing?", q_repr, [related_passage], "LOCAL")

    assert eval_res.related is True, "Passage mentions cloud computing so it should be RELATED = True"
    assert eval_res.answerable is False, "Passage does NOT define cloud computing so ANSWERABLE must be False"
    assert eval_res.decision == "LOCAL_INSUFFICIENT"


def test_cloud_computing_direct_definition_is_answerable():
    """
    Test direct definition passage MUST evaluate as:
    related = True
    answerable = True
    """
    q_repr = GenericQuestionAnalyzer.analyze("What is cloud computing?")

    def_passage = EvidenceItem(
        citation_id="E1",
        unit_id="CY024_U01",
        parent_chunk_id="CY024_C01",
        chunk_id="CY024_C01",
        paper_id="CY024",
        section_id="SEC_INTRO",
        section_name="Introduction",
        domain="cybersecurity",
        subtopic="cloud_definition",
        page_start=1,
        page_end=1,
        text="Cloud computing is defined as a model for enabling ubiquitous, convenient, on-demand network access to a shared pool of configurable computing resources including networks, servers, storage, applications, and services.",
    )

    eval_res = GenericEvidenceEvaluator.evaluate("What is cloud computing?", q_repr, [def_passage], "LOCAL")

    assert eval_res.related is True
    assert eval_res.answerable is True
    assert eval_res.decision == "LOCAL_SUFFICIENT"


def test_llm_limitations_related_vs_answerable():
    """
    Passage merely mentioning LLMs is NOT answerable for a limitations query.
    """
    q_repr = GenericQuestionAnalyzer.analyze("What are the limitations of LLMs?")

    usage_passage = EvidenceItem(
        citation_id="E1",
        unit_id="AI010_U01",
        parent_chunk_id="AI010_C01",
        chunk_id="AI010_C01",
        paper_id="AI010",
        section_id="SEC_INTRO",
        section_name="Introduction",
        domain="artificial_intelligence",
        subtopic="llm_applications",
        page_start=1,
        page_end=1,
        text="Large language models (LLMs) are widely used in image-to-text systems and automated translation pipelines across multi-modal research domains.",
    )

    eval_res = GenericEvidenceEvaluator.evaluate("What are the limitations of LLMs?", q_repr, [usage_passage], "LOCAL")

    assert eval_res.related is True
    assert eval_res.answerable is False, "Usage passage does not list limitations"


def test_llm_limitations_explicit_drawbacks_is_answerable():
    q_repr = GenericQuestionAnalyzer.analyze("What are the limitations of LLMs?")

    limitation_passage = EvidenceItem(
        citation_id="E1",
        unit_id="AI011_U01",
        parent_chunk_id="AI011_C01",
        chunk_id="AI011_C01",
        paper_id="AI011",
        section_id="SEC_LIMITATIONS",
        section_name="Limitations",
        domain="artificial_intelligence",
        subtopic="llm_drawbacks",
        page_start=8,
        page_end=9,
        text="Major limitations of LLMs include severe hallucination risk, high computational memory bottlenecks during inference, and vulnerability to adversarial jailbreak prompts.",
    )

    eval_res = GenericEvidenceEvaluator.evaluate("What are the limitations of LLMs?", q_repr, [limitation_passage], "LOCAL")

    assert eval_res.related is True
    assert eval_res.answerable is True


def test_agentic_rag_evaluation_limitations_is_unanswerable():
    """
    Test live failure case:
    Passage discussing limitations of evaluation practices for Agentic RAG MUST evaluate as:
    related = True
    answerable = False
    """
    q_repr = GenericQuestionAnalyzer.analyze("What are the limitations of agentic RAG?")

    eval_practices_passage = EvidenceItem(
        citation_id="E1",
        unit_id="AI030_U01",
        parent_chunk_id="AI030_C01",
        chunk_id="AI030_C01",
        paper_id="AI030",
        section_id="SEC_EVAL",
        section_name="Evaluation Practices",
        domain="artificial_intelligence",
        subtopic="agentic_rag_eval",
        page_start=3,
        page_end=3,
        text="This section therefore examines the limitations of existing evaluation practices and outlines a structured framework for assessing agentic RAG behavior.",
    )

    eval_res = GenericEvidenceEvaluator.evaluate("What are the limitations of agentic RAG?", q_repr, [eval_practices_passage], "LOCAL")

    assert eval_res.related is True, "Passage discusses Agentic RAG so it should be RELATED = True"
    assert eval_res.answerable is False, "Passage discusses limitations of evaluation practices, NOT limitations of Agentic RAG"
    assert eval_res.decision == "LOCAL_INSUFFICIENT"


def test_agentic_rag_evaluation_dimensions_is_unanswerable():
    q_repr = GenericQuestionAnalyzer.analyze("What are the limitations of agentic RAG?")

    eval_dim_passage = EvidenceItem(
        citation_id="E1",
        unit_id="AI030_U02",
        parent_chunk_id="AI030_C02",
        chunk_id="AI030_C02",
        paper_id="AI030",
        section_id="SEC_METRICS",
        section_name="Evaluation Dimensions",
        domain="artificial_intelligence",
        subtopic="agentic_rag_eval",
        page_start=4,
        page_end=4,
        text="Evaluation Dimensions for Agentic RAG... Faithfulness: The degree to which a generated response remains strictly aligned with the retrieved context.",
    )

    eval_res = GenericEvidenceEvaluator.evaluate("What are the limitations of agentic RAG?", q_repr, [eval_dim_passage], "LOCAL")

    assert eval_res.related is True
    assert eval_res.answerable is False, "Passage lists evaluation dimensions, NOT limitations of Agentic RAG"


def test_agentic_rag_direct_limitations_is_answerable():
    q_repr = GenericQuestionAnalyzer.analyze("What are the limitations of agentic RAG?")

    direct_limitation_passage = EvidenceItem(
        citation_id="E1",
        unit_id="AI031_U01",
        parent_chunk_id="AI031_C01",
        chunk_id="AI031_C01",
        paper_id="AI031",
        section_id="SEC_LIMITATIONS",
        section_name="Limitations",
        domain="artificial_intelligence",
        subtopic="agentic_rag_drawbacks",
        page_start=5,
        page_end=6,
        text="Agentic RAG systems can suffer from excessive tool calls, inefficient planning, error propagation across agent steps, retrieval failures, and increased computational cost.",
    )

    eval_res = GenericEvidenceEvaluator.evaluate("What are the limitations of agentic RAG?", q_repr, [direct_limitation_passage], "LOCAL")

    assert eval_res.related is True
    assert eval_res.answerable is True
    assert eval_res.decision == "LOCAL_SUFFICIENT"

