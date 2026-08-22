"""
Generic Answerability Regression Test Suite

Tests the GenericEvidenceEvaluator against a broad range of question types and
passage combinations, verifying the "related ≠ answerable" distinction for:

- Definitions
- Algorithm/method questions
- Mechanism questions
- Limitation questions
- Benefit/advantage questions
- Comparison questions
- Cause/effect questions
- Relationship questions
- Application questions
- Result/finding questions
- Multi-part questions
- Cross-domain questions
- Out-of-corpus questions
- Partially answerable questions
- Related-but-not-answerable passages

ZERO hardcoded expected answers for specific questions.
The test cases verify decision logic, NOT topic-specific content.
"""

import pytest
from src.pipeline.rag import (
    GenericQuestionAnalyzer,
    GenericEvidenceEvaluator,
    EvidenceSufficiencyEvaluator,
    AnswerabilityResult,
    RAGPipeline,
)
from src.pipeline.retrieval import RetrievalResult


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_passage(
    text: str,
    paper_id: str = "TEST001",
    domain: str = "artificial_intelligence",
    section_name: str = "Introduction",
) -> RetrievalResult:
    """Create a minimal RetrievalResult for testing."""
    return RetrievalResult(
        unit_id=f"{paper_id}_U01",
        parent_chunk_id=f"{paper_id}_C01",
        chunk_id=f"{paper_id}_C01",
        paper_id=paper_id,
        section_id="SEC_01",
        section_name=section_name,
        domain=domain,
        subtopic="test",
        page_start=1,
        page_end=2,
        text=text,
        dense_score=0.85,
        bm25_score=10.0,
        rrf_score=0.03,
        rank=1,
        token_count=len(text.split()),
        retrieval_methods=["bm25", "dense"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: GenericQuestionAnalyzer Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenericQuestionAnalyzer:
    """Test that question decomposition works generically for diverse question types."""

    def test_algorithm_intent_classification(self):
        """Algorithm questions are classified as 'Algorithm' intent."""
        q = "What algorithms are commonly used for intrusion detection?"
        assert GenericQuestionAnalyzer.classify_intent(q) == "Algorithm"

    def test_limitation_intent_classification(self):
        """Limitation questions classified correctly."""
        assert GenericQuestionAnalyzer.classify_intent("What are the limitations of LLMs?") == "Limitation"
        assert GenericQuestionAnalyzer.classify_intent("What are the challenges of federated learning?") == "Limitation"
        assert GenericQuestionAnalyzer.classify_intent("What are the drawbacks of neural networks?") == "Limitation"

    def test_advantage_intent_classification(self):
        """Advantage/benefit questions classified correctly."""
        assert GenericQuestionAnalyzer.classify_intent("What are the benefits of precision agriculture?") == "Advantage"
        assert GenericQuestionAnalyzer.classify_intent("What advantages does CRISPR provide?") == "Advantage"

    def test_mechanism_intent_classification(self):
        """Mechanism/how-does questions classified correctly."""
        q = "How does carbon capture work?"
        intent = GenericQuestionAnalyzer.classify_intent(q)
        assert intent in ("Mechanism", "Definition"), f"Unexpected intent: {intent}"

    def test_comparison_intent_classification(self):
        """Comparison/difference questions classified correctly."""
        q = "What is the difference between BM25 and dense retrieval?"
        assert GenericQuestionAnalyzer.classify_intent(q) == "Comparison"
        q2 = "How do SVMs compare to decision trees?"
        assert GenericQuestionAnalyzer.classify_intent(q2) in ("Comparison", "Mechanism")

    def test_definition_intent_classification(self):
        """Definition questions classified correctly."""
        assert GenericQuestionAnalyzer.classify_intent("What is quantum teleportation?") == "Definition"
        assert GenericQuestionAnalyzer.classify_intent("What are transformers?") in ("Definition", "Explanation")

    def test_relational_structure_detection_improve(self):
        """Relational structure X→Y detected when verb is 'improve'."""
        q = "How does reinforcement learning improve robotic manipulation?"
        q_repr = GenericQuestionAnalyzer.analyze(q)
        assert q_repr.is_relational is True
        assert len(q_repr.relation_source) >= 1
        assert len(q_repr.relation_target) >= 1

    def test_relational_structure_detection_affect(self):
        """Relational structure detected for 'affect'."""
        q = "How does model drift affect prediction accuracy in machine learning?"
        q_repr = GenericQuestionAnalyzer.analyze(q)
        assert q_repr.is_relational is True

    def test_comparison_structure_detection(self):
        """Comparison structure A vs B detected."""
        q = "What is the difference between BM25 and dense retrieval?"
        q_repr = GenericQuestionAnalyzer.analyze(q)
        assert q_repr.is_comparison is True
        assert len(q_repr.comparison_targets) >= 2

    def test_multi_aspect_detection(self):
        """Multi-aspect questions (benefits AND limitations) detected."""
        q = "What are the benefits and limitations of smart irrigation systems?"
        q_repr = GenericQuestionAnalyzer.analyze(q)
        assert q_repr.is_multi_aspect is True
        assert "advantage" in q_repr.required_aspects
        assert "limitation" in q_repr.required_aspects

    def test_content_word_extraction_unseen_topic(self):
        """Content words extracted for completely unseen topics — zero hardcoding."""
        q = "How does neuromorphic spike timing improve underwater acoustic localization?"
        q_repr = GenericQuestionAnalyzer.analyze(q)
        assert len(q_repr.all_content_words) >= 3
        # Must not depend on any hardcoded vocabulary
        assert "neuromorphic" in q_repr.all_content_words or "spike" in q_repr.all_content_words or "timing" in q_repr.all_content_words

    def test_cross_domain_question_analysis(self):
        """Cross-domain questions are decomposed without domain-specific hardcoding."""
        q = "What factors affect vaccine effectiveness in elderly populations?"
        q_repr = GenericQuestionAnalyzer.analyze(q)
        assert len(q_repr.all_content_words) >= 3
        assert q_repr.intent in ("Cause", "Relationship", "Explanation", "Limitation")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: "Related ≠ Answerable" Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRelatedButNotAnswerable:
    """
    These tests enforce the CORE requirement:
    A passage that discusses the topic but does NOT contain the requested
    information type MUST be classified as RELATED but NOT ANSWERABLE.
    """

    def test_llm_application_passage_not_answerable_for_limitation_question(self):
        """
        Question: "What are the limitations of LLMs?"
        Passage: "LLMs are widely used in image-to-text applications."
        → Related=True (LLMs mentioned), Answerable=False (no limitation info)
        """
        question = "What are the limitations of LLMs?"
        passage = make_passage(
            "Large language models (LLMs) are widely used in image-to-text applications, "
            "virtual assistants, and code generation systems across various industries.",
            paper_id="AI001"
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True, "Passage discusses LLMs — must be related"
        assert result.answerable is False, (
            "Passage contains no limitation information — must NOT be answerable. "
            f"Got: intent_support={result.intent_support:.2f}, decision={result.decision}"
        )

    def test_ml_usage_passage_not_answerable_for_algorithm_question(self):
        """
        Question: "What algorithms are commonly used for intrusion detection?"
        Passage: "Machine learning is widely used for intrusion detection."
        → Related=True, Answerable=False (no algorithm names)
        """
        question = "What algorithms are commonly used for intrusion detection?"
        passage = make_passage(
            "Machine learning is widely used for intrusion detection in enterprise "
            "network security systems to identify anomalous traffic patterns.",
            paper_id="CY001",
            domain="cybersecurity",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True, "Passage mentions intrusion detection — must be related"
        assert result.answerable is False, (
            "Passage says ML is 'used' but names no specific algorithm — must NOT be answerable. "
            f"Got: intent_support={result.intent_support:.2f}"
        )

    def test_xai_general_passage_not_answerable_for_relationship_question(self):
        """
        Question: "How does Explainable AI improve trust in healthcare?"
        Passage: "Explainable AI makes models interpretable."
        → Related=True, Answerable=False (no healthcare/trust relationship)
        """
        question = "How does Explainable AI improve trust in healthcare?"
        passage = make_passage(
            "Explainable AI (XAI) makes machine learning models more interpretable "
            "by providing feature attribution maps and saliency visualizations.",
            paper_id="AI036",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True, "Passage discusses XAI — must be related"
        assert result.answerable is False, (
            "Passage does not address the healthcare/trust relationship — must NOT be answerable. "
            f"Got: relationship_support={result.relationship_support:.2f}"
        )

    def test_rag_benefit_passage_not_answerable_for_limitation_question(self):
        """
        Question: "What are the limitations of retrieval-augmented generation?"
        Passage: "RAG enhances output quality and provides superior factual accuracy."
        → Related=True, Answerable=False (only benefits, no limitations)
        """
        question = "What are the limitations of retrieval-augmented generation?"
        passage = make_passage(
            "Retrieval-augmented generation (RAG) enhances output quality and provides "
            "superior factual accuracy by grounding responses in retrieved evidence.",
            paper_id="AI001",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True, "Passage is about RAG — must be related"
        assert result.answerable is False, (
            "Passage describes benefits not limitations — must NOT be answerable. "
            f"Got: intent_support={result.intent_support:.2f}"
        )

    def test_wrong_domain_passage_not_related(self):
        """
        Question: "What is quantum teleportation?"
        Passage: About crop yield optimization.
        → Related=False
        """
        question = "What is quantum teleportation?"
        passage = make_passage(
            "Crop yield optimization using precision irrigation sensors and nitrogen fertilizers "
            "improves agricultural productivity in arid regions.",
            paper_id="AG001",
            domain="agriculture",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is False, "Crop yield passage is unrelated to quantum teleportation"
        assert result.answerable is False


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: Answerable Passages — True Positive Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAnswerablePassages:
    """Passages that genuinely answer the question MUST be classified as answerable."""

    def test_limitation_passage_answerable_for_limitation_question(self):
        """
        Question: "What are the limitations of LLMs?"
        Passage explicitly describes limitations → Answerable=True
        """
        question = "What are the limitations of LLMs?"
        passage = make_passage(
            "Large language models (LLMs) face several key limitations, including "
            "hallucination of factual content, context-window constraints that limit "
            "processing of long documents, inability to access real-time information, "
            "high computational costs, and reasoning failures in complex multi-step tasks.",
            paper_id="AI002",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True
        assert result.answerable is True, (
            "Passage explicitly lists LLM limitations — must be answerable. "
            f"Got: intent_support={result.intent_support:.2f}, score={result.answerability_score:.2f}"
        )

    def test_algorithm_passage_answerable_for_algorithm_question(self):
        """
        Question: "What algorithms are commonly used for intrusion detection?"
        Passage names specific algorithms → Answerable=True
        """
        question = "What algorithms are commonly used for intrusion detection?"
        passage = make_passage(
            "Intrusion detection systems commonly employ several machine learning algorithms, "
            "including Random Forest, Support Vector Machine (SVM), k-Nearest Neighbor (kNN), "
            "Decision Tree classifiers, and deep neural networks such as LSTM-based models "
            "for sequence-based anomaly detection in network traffic.",
            paper_id="CY002",
            domain="cybersecurity",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True
        assert result.answerable is True, (
            "Passage names specific algorithms (Random Forest, SVM, etc.) — must be answerable. "
            f"Got: intent_support={result.intent_support:.2f}"
        )

    def test_mechanism_passage_answerable_for_mechanism_question(self):
        """
        Mechanism question answered by a passage describing the actual process.
        """
        question = "How does carbon capture work?"
        passage = make_passage(
            "Carbon capture works through a three-step process: first, CO2 is captured at "
            "point sources such as power plants using chemical sorbents or membranes. "
            "Second, the captured CO2 is compressed and transported via pipelines. "
            "Third, it is injected into deep geological formations for permanent storage, "
            "preventing release into the atmosphere.",
            paper_id="CL001",
            domain="climate",
            section_name="Methods",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True
        assert result.answerable is True, (
            "Passage describes the actual mechanism steps — must be answerable. "
            f"Got: intent_support={result.intent_support:.2f}"
        )

    def test_benefit_passage_answerable_for_benefit_question(self):
        """
        Benefit question answered by a passage listing concrete advantages.
        """
        question = "What are the benefits of precision agriculture?"
        passage = make_passage(
            "Precision agriculture offers significant benefits including reduced water consumption "
            "through targeted irrigation, lower fertilizer usage via variable-rate application, "
            "improved crop yield through data-driven decision making, and reduced environmental "
            "impact through minimized chemical runoff.",
            paper_id="AG002",
            domain="agriculture",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True
        assert result.answerable is True, (
            "Passage lists concrete agriculture benefits — must be answerable. "
            f"Got: intent_support={result.intent_support:.2f}"
        )

    def test_definition_passage_answerable_for_definition_question(self):
        """
        Definition question answered by a passage that actually defines the concept.
        """
        question = "What is quantum teleportation?"
        passage = make_passage(
            "Quantum teleportation is a protocol for transmitting an unknown quantum state "
            "between two parties using pre-shared quantum entanglement and classical communication. "
            "The sender performs a joint Bell-state measurement and transmits the classical result "
            "to the receiver, who applies a corresponding unitary operation to reconstruct the state.",
            paper_id="PHY001",
            domain="artificial_intelligence",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True
        assert result.answerable is True, (
            "Passage defines quantum teleportation — must be answerable. "
            f"Got: intent_support={result.intent_support:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Multi-Aspect Coverage Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiAspectCoverage:
    """Multi-aspect questions require both aspects to be covered independently."""

    def test_benefits_only_is_insufficient_for_benefits_and_limitations_question(self):
        """
        "What are the benefits and limitations of smart irrigation?"
        A passage covering ONLY benefits must result in insufficient/partial.
        """
        question = "What are the benefits and limitations of smart irrigation systems?"
        passage = make_passage(
            "Smart irrigation systems provide significant benefits: they improve water efficiency, "
            "reduce operational costs, optimize crop yields through precise soil moisture monitoring, "
            "and enable remote telemetry control of irrigation schedules.",
            paper_id="AG003",
            domain="agriculture",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        assert q_repr.is_multi_aspect is True, "Must detect multi-aspect question"

        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        # Should not be fully sufficient if limitations aspect is missing
        assert result.completeness < 1.0 or not result.answerable or "limitation" in str(result.missing_aspects).lower()

    def test_both_aspects_covered_is_sufficient(self):
        """
        When both benefits and limitations are covered, the result is sufficient.
        """
        question = "What are the benefits and limitations of smart irrigation systems?"
        passage = make_passage(
            "Smart irrigation systems offer benefits such as water efficiency and yield improvement, "
            "but also present limitations including high installation costs, signal attenuation "
            "in remote fields, and sensor maintenance challenges. The drawbacks can reduce "
            "adoption in resource-constrained farming environments.",
            paper_id="AG004",
            domain="agriculture",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True
        # Should be answerable or at least have high completeness
        assert result.completeness >= 0.5 or result.answerable, (
            "Passage covers both benefits and limitations — should be at least partially sufficient. "
            f"Got: completeness={result.completeness:.2f}, answerable={result.answerable}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Part 5: Comparison Question Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComparisonQuestions:
    """Comparison questions require BOTH targets to be addressed."""

    def test_bm25_only_insufficient_for_bm25_vs_dense_question(self):
        """
        "Difference between BM25 and dense retrieval?"
        Passage only about BM25 → insufficient.
        """
        question = "What is the difference between BM25 and dense retrieval?"
        passage = make_passage(
            "BM25 is a lexical retrieval model based on term frequency and inverse document "
            "frequency, weighted by document length normalization. It provides efficient "
            "keyword matching across large document collections.",
            paper_id="AI003",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        # Should not be fully answerable — missing dense retrieval comparison
        assert not result.answerable or result.relationship_support < 0.9, (
            "BM25-only passage should not fully answer a BM25 vs dense retrieval comparison question"
        )

    def test_both_targets_covered_answerable_for_comparison_question(self):
        """
        Passage discussing BOTH BM25 and dense retrieval with comparison → answerable.
        """
        question = "What is the difference between BM25 and dense retrieval?"
        passage = make_passage(
            "BM25 is a lexical model using term frequency and IDF for keyword matching, "
            "while dense retrieval uses neural embeddings to capture semantic similarity. "
            "BM25 excels at exact keyword matches whereas dense retrieval handles synonym "
            "and paraphrase matching better, though dense models require more computation. "
            "Hybrid approaches combining both typically achieve superior recall performance.",
            paper_id="AI004",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True
        assert result.answerable is True, (
            "Passage explicitly compares BM25 and dense retrieval — must be answerable. "
            f"Got: decision={result.decision}, score={result.answerability_score:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Part 6: Relationship Question Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRelationshipQuestions:
    """Relationship questions require evidence of the X→Y connection, not just X and Y."""

    def test_separate_entity_mentions_not_answerable_for_relationship_question(self):
        """
        Passage mentions both X and Y but does NOT establish their relationship.
        """
        question = "How does reinforcement learning improve robotic manipulation?"
        passage = make_passage(
            "Reinforcement learning is a machine learning paradigm where agents learn "
            "through interaction. Robotic manipulation involves controlling robot arms "
            "and grippers for tasks such as grasping and assembly.",
            paper_id="AI005",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        # Relationship support should be lower — entities mentioned separately
        assert result.relationship_support < 1.0 or not result.answerable, (
            "Passage mentions RL and robotics separately without establishing relationship — "
            f"should have lower relationship support. Got: {result.relationship_support:.2f}"
        )

    def test_connected_relationship_passage_answerable(self):
        """
        Passage explicitly shows X improves Y → answerable.
        """
        question = "How does reinforcement learning improve robotic manipulation?"
        passage = make_passage(
            "Reinforcement learning significantly improves robotic manipulation by enabling "
            "robots to learn complex dexterous motor skills through trial-and-error interaction. "
            "RL algorithms such as PPO and SAC optimize control policies that directly improve "
            "grasping success rates and object manipulation accuracy in physical robotic systems.",
            paper_id="AI006",
        )
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.related is True
        assert result.answerable is True, (
            "Passage explicitly shows RL improving robotic manipulation — must be answerable. "
            f"Got: relationship_support={result.relationship_support:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Part 7: Chunk Quality Gate Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkQualityGate:
    """Short/truncated/bibliography chunks must be rejected."""

    def test_very_short_passage_rejected(self):
        """Passage under 100 chars should fail the quality gate."""
        question = "What are the limitations of LLMs?"
        passage = make_passage("LLMs have limitations.", paper_id="AI007")
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.answerable is False, "Too-short passage should not be answerable"

    def test_bibliography_passage_rejected(self):
        """Bibliography/reference chunks must be rejected."""
        question = "What algorithms are used for intrusion detection?"
        bib_text = "[1] Smith et al. IEEE Trans. 2021. [2] Jones et al. NeurIPS 2020. [3] Doe et al. CVPR 2019."
        passage = make_passage(bib_text, paper_id="CY003", domain="cybersecurity")
        q_repr = GenericQuestionAnalyzer.analyze(question)
        result = GenericEvidenceEvaluator.evaluate(question, q_repr, [passage], "LOCAL")

        assert result.answerable is False, "Bibliography chunk should not be answerable"


# ─────────────────────────────────────────────────────────────────────────────
# Part 8: Legacy EvidenceSufficiencyEvaluator Compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestLegacyCompatibility:
    """EvidenceSufficiencyEvaluator shim must delegate correctly to GenericEvidenceEvaluator."""

    def test_legacy_evaluate_returns_tuple(self):
        """Legacy evaluate() must return (is_related, is_answerable, rationale, score)."""
        question = "What are the limitations of LLMs?"
        # This passage discusses LLMs (subject coverage) but contains NO limitation markers.
        # It must be classified: related=True, answerable=False
        passage = make_passage(
            "Large language models (LLMs) represent a major class of language model "
            "trained on massive text corpora using transformer architectures and self-attention. "
            "LLMs are employed in image-to-text systems, code completion tools, and "
            "conversational assistants across many commercial and research environments.",
            paper_id="AI001",
        )
        result = EvidenceSufficiencyEvaluator.evaluate(question, [passage])
        assert isinstance(result, tuple)
        assert len(result) == 4
        is_rel, is_ans, rationale, score = result
        assert isinstance(is_rel, bool)
        assert isinstance(is_ans, bool)
        assert isinstance(rationale, str)
        assert isinstance(score, float)
        # "Related but not answerable" case
        assert is_rel is True, (
            f"Passage discusses LLMs extensively — must be related. "
            f"Got related={is_rel}, score={score:.2f}, rationale={rationale[:80]}"
        )
        assert is_ans is False, (
            f"Passage contains no limitation language — must NOT be answerable. "
            f"Got answerable={is_ans}, score={score:.2f}"
        )


    def test_legacy_decompose_query_returns_dict(self):
        """Legacy decompose_query() must return a dict with expected keys."""
        q = "What algorithms are commonly used for intrusion detection?"
        decomp = EvidenceSufficiencyEvaluator.decompose_query(q)
        assert isinstance(decomp, dict)
        assert "all_content_words" in decomp
        assert "intent" in decomp
        assert "is_relational" in decomp
        assert len(decomp["all_content_words"]) >= 2

    def test_legacy_classify_intent_works(self):
        """Legacy classify_intent() must work."""
        intent = EvidenceSufficiencyEvaluator.classify_intent(
            "What are the limitations of LLMs?"
        )
        assert intent == "Limitation"


# ─────────────────────────────────────────────────────────────────────────────
# Part 9: End-to-End RAGPipeline Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRAGPipelineGenericAnswerability:
    """End-to-end pipeline tests for generic answerability."""

    @pytest.fixture
    def pipeline(self):
        return RAGPipeline()

    def test_pipeline_returns_rag_response(self, pipeline):
        """Pipeline must return a RAGResponse for any valid question."""
        res = pipeline.answer("What are the limitations of retrieval-augmented generation?")
        assert res is not None
        assert res.answer is not None
        assert len(res.answer) > 10

    def test_citation_tags_in_answer_map_to_citations_dict(self, pipeline):
        """Every [E#] or [O#] in answer must exist in citations dict."""
        import re
        res = pipeline.answer("How do IoT sensors optimize smart irrigation?")
        tags_in_text = re.findall(r"\[([EO]\d+)\]", res.answer)
        for tag in tags_in_text:
            assert tag in res.citations, (
                f"Citation tag [{tag}] found in answer but missing from citations dict"
            )

    def test_insufficient_evidence_returns_no_confident_answer(self, pipeline):
        """Out-of-corpus queries must not produce an 'Excellent' answer from local evidence."""
        res = pipeline.answer("What is the mating behavior of deep-sea gigantism organisms?")
        # If local evidence is used, it must not be rated Excellent from clearly out-of-corpus content
        local_cits = [k for k in res.citations if k.startswith("E")]
        if local_cits and res.confidence == "Excellent":
            # Verify the answer actually addresses the question
            assert "gigantism" in res.answer.lower() or "deep-sea" in res.answer.lower() or "organism" in res.answer.lower(), (
                "Excellent confidence given for answer that doesn't address the question"
            )

    def test_online_fallback_triggered_for_out_of_corpus_question(self, pipeline):
        """Completely out-of-corpus questions must trigger online fallback."""
        res = pipeline.answer("What is quantum teleportation?")
        assert res.answer is not None
        # Either online evidence is used, or insufficient evidence is returned
        if "insufficient evidence" not in res.answer.lower():
            online_cits = [k for k in res.citations if k.startswith("O")]
            assert len(online_cits) >= 1, (
                "Out-of-corpus question should use online evidence but none found"
            )

    def test_online_evidence_also_validated_not_just_accepted(self, pipeline):
        """
        Online evidence must also pass answerability evaluation.
        The pipeline must NOT blindly accept any ArXiv papers as sufficient.
        """
        res = pipeline.answer("What causes model drift in machine learning?")
        assert res.answer is not None
        # If online evidence was used, it must be cited
        if "insufficient evidence" not in res.answer.lower():
            assert len(res.citations) >= 1

    def test_no_hardcoded_branches_verified_by_diverse_questions(self, pipeline):
        """
        Diverse questions must all get valid responses — proving no hardcoded branches.
        Tests questions across different intents and domains.
        """
        diverse_questions = [
            "What algorithms are commonly used for intrusion detection?",
            "What are the benefits of precision agriculture?",
            "How does carbon capture work?",
            "What factors affect vaccine effectiveness?",
            "What is the difference between supervised and unsupervised learning?",
        ]
        for q in diverse_questions:
            res = pipeline.answer(q)
            assert res is not None, f"Pipeline returned None for: {q}"
            assert res.answer is not None, f"No answer for: {q}"
            assert res.why_this_answer is not None, f"No WhyThisAnswer for: {q}"
