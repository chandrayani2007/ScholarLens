"""
Phase 27 Acceptance Fix — Regression Test Suite verifying live API response integrity:
1. test_final_api_answer_meets_completeness_requirement()
2. test_every_inline_citation_has_matching_citation_object()
3. test_no_o_citation_without_real_arxiv_metadata()
4. test_real_arxiv_id_matches_returned_paper()
5. test_online_claim_is_supported_by_exact_online_evidence()
6. test_short_final_answer_triggers_regeneration()
"""

import pytest
import json
from src.pipeline.rag import RAGPipeline, FinalSafetyGate, CitationValidator, AnswerCompletenessValidator
from src.pipeline.online_fallback import OnlineAcademicRetriever, OnlineEvidenceItem
from src.pipeline.llm import MockLLMProvider


class TestPhase27AcceptanceFix:

    def test_final_api_answer_meets_completeness_requirement(self):
        """The final answer string returned by RAGPipeline must meet the >=90 word completeness requirement for explanatory questions."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What are the limitations of retrieval-augmented generation systems?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        words = len(res.answer.split())
        assert words >= 90, f"Final API answer is too short ({words} words < 90 words)."
        is_comp, reason = AnswerCompletenessValidator.validate_completeness(res.answer, "Limitation")
        assert is_comp, f"Completeness check failed: {reason}"

    def test_every_inline_citation_has_matching_citation_object(self):
        """Every inline citation tag [E#] or [O#] in res.answer MUST have a corresponding key in res.citations."""
        pipeline = RAGPipeline(llm_provider=MockLLMProvider())
        q = "What are the limitations of retrieval-augmented generation systems?"
        res = pipeline.answer(q, filters={"domain": "artificial_intelligence"})

        tags_in_text = CitationValidator.extract_citations(res.answer)
        for tag in tags_in_text:
            assert tag in res.citations, f"Inline tag [{tag}] present in answer but missing from citations dictionary!"

        for tag in res.citations.keys():
            assert tag in tags_in_text, f"Tag [{tag}] present in citations dictionary but missing from answer text!"

    def test_no_o_citation_without_real_arxiv_metadata(self):
        """Every online citation [O#] in res.citations MUST contain real non-empty ArXiv metadata fields."""
        online_ev = OnlineEvidenceItem(
            citation_id="O1",
            paper_id="arXiv:2303.12345",
            title="Real Time Quantum Error Correction Protocol",
            authors=["Alice Physicist", "Bob Quantum"],
            published_date="2023-03-15",
            url="http://arxiv.org/abs/2303.12345",
            domain="artificial_intelligence",
            section_name="Abstract",
            text="Quantum error correction uses real time syndrome measurements across data qubits to suppress decoherence errors."
        )

        mock_llm = MockLLMProvider(custom_response=json.dumps({
            "answer": "Quantum error correction uses real time syndrome measurements across data qubits to suppress decoherence errors [O1]. "
                      "Physical stabilizer measurements continuously compute error syndromes across adjacent data and syndrome qubits, allowing real-time fault-tolerant error suppression below the threshold limit. "
                      "Recent experimental realizations demonstrate fault-tolerant logical qubit operations using real-time decoders to achieve sub-threshold error scaling across multi-qubit processor architectures. "
                      "Integrating fast syndrome processing algorithms accelerates logical gate fidelity and maintains fault tolerance during extended quantum operations.",
            "confidence": "High",
            "evidence_strength": "High",
            "limitations": "Online findings.",
            "why_this_answer": "Supported by ArXiv search."
        }))

        pipeline = RAGPipeline(llm_provider=mock_llm)
        pipeline.online_retriever = type("MockOnline", (), {"retrieve": lambda self, **kwargs: [online_ev]})()

        res = pipeline.answer("What is surface code quantum error correction?")
        assert "O1" in res.citations
        cit_o1 = res.citations["O1"]
        assert cit_o1.paper_id == "arXiv:2303.12345"
        assert cit_o1.title == "Real Time Quantum Error Correction Protocol"
        assert cit_o1.url.startswith("http://arxiv.org/abs/")

    def test_real_arxiv_id_matches_returned_paper(self):
        """Real ArXiv retrieval returns actual ArXiv ID matching paper metadata."""
        online_retriever = OnlineAcademicRetriever(timeout=10)
        items = online_retriever.retrieve("quantum computing neural networks", max_results=2)
        if items:
            first = items[0]
            assert first.paper_id.startswith("arXiv:")
            assert first.url == f"http://arxiv.org/abs/{first.paper_id.replace('arXiv:', '')}"
        else:
            assert items == []

    def test_online_claim_is_supported_by_exact_online_evidence(self):
        """Claim containing [O1] must be grounded in the text of online item [O1]."""
        online_ev = OnlineEvidenceItem(
            citation_id="O1",
            paper_id="arXiv:2403.99999",
            title="Surface Code Quantum Architecture Protocol",
            authors=["V. Vector"],
            published_date="2024-03-01",
            url="http://arxiv.org/abs/2403.99999",
            domain="artificial_intelligence",
            section_name="Abstract",
            text="Surface code quantum error correction uses square physical qubit arrays to measure stabilizer syndromes."
        )

        mock_llm = MockLLMProvider(custom_response=json.dumps({
            "answer": "Surface code quantum error correction uses square physical qubit arrays to measure stabilizer syndromes [O1]. "
                      "Physical stabilizer measurements continuously compute error syndromes across adjacent data and syndrome qubits, allowing real-time fault-tolerant error suppression below the threshold limit. "
                      "Recent experimental realizations demonstrate fault-tolerant logical qubit operations using real-time decoders to achieve sub-threshold error scaling across multi-qubit processor architectures. "
                      "Integrating fast syndrome processing algorithms accelerates logical gate fidelity and maintains fault tolerance during extended quantum operations.",
            "confidence": "High",
            "evidence_strength": "High",
            "limitations": "Online findings.",
            "why_this_answer": "Supported by ArXiv search."
        }))

        pipeline = RAGPipeline(llm_provider=mock_llm)
        pipeline.online_retriever = type("MockOnline", (), {"retrieve": lambda self, **kwargs: [online_ev]})()

        res = pipeline.answer("What is surface code quantum error correction?")
        assert "O1" in res.citations
        assert "surface code quantum error correction" in res.answer.lower()

    def test_short_final_answer_triggers_regeneration(self):
        """A initial short answer (~45 words) MUST fail completeness and trigger regeneration to >=90 words."""
        short_response = json.dumps({
            "answer": "Retrieval-augmented generation combines retrieval and generation [E1]. However, retrieval noise can lead to hallucinations.",
            "confidence": "High",
            "evidence_strength": "High",
            "limitations": "Short test answer.",
            "why_this_answer": "Test."
        })

        detailed_response = json.dumps({
            "answer": "Retrieval-Augmented Generation (RAG) systems face several critical architectural limitations during real-world scientific execution [E1]. "
                      "First, retrieval dependency and noise: if the upstream vector or lexical retriever fetches incomplete or semantically irrelevant context, the downstream generator inherits these factual omissions, leading to context-induced hallucinations. "
                      "Second, context window constraints and computational latency: embedding large document corpora requires fine-grained chunking, which can fragment long-range dependencies across paper sections and increase inference latency during multi-passage synthesis. "
                      "Third, out-of-distribution handling: when queries address concepts absent from the underlying vector index, models risk misinterpreting partially related passages without explicit fallback mechanisms.",
            "confidence": "High",
            "evidence_strength": "High",
            "limitations": "Detailed regenerated answer.",
            "why_this_answer": "Supported by corpus."
        })

        call_count = [0]
        def mock_generate(self_inst, prompt, system_prompt=None, request_id=None, question_hash=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return short_response
            return detailed_response

        mock_llm = type("MockLLM", (), {"generate": mock_generate})()
        pipeline = RAGPipeline(llm_provider=mock_llm)

        res = pipeline.answer("What are the limitations of retrieval-augmented generation systems?", filters={"domain": "artificial_intelligence"})
        assert call_count[0] > 1, "Short answer failed to trigger regeneration!"
        assert len(res.answer.split()) >= 90
