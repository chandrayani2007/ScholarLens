import json
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.rag import RAGPipeline, AntiCopyValidator, GenericQuestionAnalyzer, EvidenceCoverageTracker
from src.pipeline.llm import get_llm_provider, GeminiLLMProvider, MockLLMProvider

OPEN_SCHOLAR_PAPER_TEXT = """
Title: OpenScholar: Synthesizing Scientific Literature with Open Retrieval-Augmented Language Models

1 Introduction & Main Research Problem
Synthesizing knowledge from the scientific literature is essential for discovering new directions, refining methodologies and supporting evidence-based decisions, yet the rapid growth of publications makes it increasingly difficult for researchers to stay informed. Effective synthesis requires precise retrieval, accurate attribution and access to up-to-date literature. General-purpose language models (LLMs) can assist with synthesis but often suffer from factual hallucinations, outdated pre-training knowledge and insufficient citation grounding. The main research problem addressed by this paper is the challenge of accurately synthesizing reliable scientific knowledge across voluminous academic publications while eliminating model hallucinations and ensuring verifiable citation grounding.

2 Proposed Methodology & System Architecture
We introduce OpenScholar, an open scientific literature synthesis framework designed for transparent, evidence-grounded scientific question answering. OpenScholar couples an academic passage retriever trained on scientific corpora with a specialized 8B parameter language model and an iterative self-feedback inference mechanism, termed OpenScholar Feedback Loop (OSDS). The system retrieves relevant passages from a specialized scientific data store and dynamically critiques and refines draft responses to maximize factual accuracy and attribution.

3 Dataset, Benchmark & Data Preparation
To rigorously evaluate literature synthesis systems, we present ScholarQABench, an evaluation suite covering four core disciplines: computer science, physics, biomedicine and neuroscience. ScholarQABench comprises two main subsets: Scholar-CS, containing 100 questions formulated by PhD candidates, and Scholar-Multi, containing 108 challenging questions paired with 250 expert-written long-form answers. The benchmark queries and reference answers were formulated, prepared, and preprocessed by experienced PhD candidates and postdoctoral scholars to accurately capture realistic scientific literature review workflows.

4 Main Quantitative Results & Baseline Comparison
Experimental evaluations on ScholarQABench demonstrate that OpenScholar models achieve substantial quantitative improvements over unassisted baselines. OpenScholar-GPT-4o improves correctness by 12% over base GPT-4o. In expert blind evaluations conducted with 16 PhD researchers, OpenScholar-8B and OpenScholar-GPT-4o responses were preferred over expert-written human answers 51% and 70% of the time, respectively, whereas baseline GPT-4o achieved only a 32% win rate.

5 Main Contributions
The authors make three primary contributions: (1) creating OpenScholar, an open literature synthesis framework with iterative self-critique inference, (2) establishing ScholarQABench across four scientific disciplines with 250 expert-written answers, and (3) showing that retrieval-augmented feedback substantially boosts correctness and human preference over foundation baselines.

6 Author-Stated Limitations & Discussion
The authors observed several key limitations in current literature synthesis models. Standard language models without retrieval frequently fabricate citations and struggle with multi-paper reference attribution. Additionally, executing iterative self-feedback critique loops introduces computational latency during long-form answer generation. Future work aims to scale datastore indexing and accelerate multi-step reasoning.
"""

def main():
    llm_provider = get_llm_provider()
    print(f"INITIALIZED LLM PROVIDER: {type(llm_provider).__name__}")
    pipeline = RAGPipeline(llm_provider=llm_provider)

    test_questions = [
        "What is the research problem addressed by this paper?",
        "What is the proposed methodology of OpenScholar?",
        "What datasets and benchmarks were used for evaluation?",
        "Was the data prepared or preprocessed before evaluation?",
        "What experimental setup was used, including models, baselines and evaluation procedure?",
        "What are the quantitative results of OpenScholar?",
        "How does OpenScholar compare with the baseline methods?",
        "What are the main scientific and technical contributions of this paper?",
        "What limitations do the authors identify?",
        "Give a comprehensive summary covering the research problem, methodology, datasets, experimental setup, results, contributions and limitations."
    ]

    results = []

    print("\n" + "="*80)
    print("RUNNING 10 EVALUATION QUESTIONS ON SDC_BasePaper.pdf")
    print("="*80 + "\n")

    for idx, q in enumerate(test_questions, 1):
        print(f"\n{'-'*80}")
        print(f"[{idx}/10] TESTING QUESTION: {q}")
        print(f"{'-'*80}")

        response = pipeline.answer(
            q,
            uploaded_paper_text=OPEN_SCHOLAR_PAPER_TEXT,
            uploaded_paper_name="SDC_BasePaper.pdf",
            filters={"paper_id": "uploaded"}
        )

        answer = response.answer
        citations = list(response.citations.keys())
        why = response.why_this_answer

        # Analyze coverage
        q_repr = GenericQuestionAnalyzer.analyze(q)
        cov_items, cov_state, cov_ratio, missing = EvidenceCoverageTracker.evaluate_coverage(
            q, q_repr, response.evidence
        )

        # Anti-copy check
        is_syn, reason, score = AntiCopyValidator.validate_answer(answer, response.evidence)

        # Basic quality checks
        has_citations = len(citations) > 0 or "insufficient" in answer.lower()
        non_empty = len(answer.strip()) > 50

        passed = non_empty and is_syn

        results.append({
            "num": idx,
            "question": q,
            "intent": q_repr.intent,
            "q_type": q_repr.question_type,
            "is_synthesized": is_syn,
            "overlap_score": score,
            "coverage_state": cov_state,
            "coverage_ratio": cov_ratio,
            "missing_aspects": missing,
            "citations": citations,
            "evidence_strength": why.evidence_strength if why else "N/A",
            "passed": passed,
            "answer_preview": answer[:300] + "..." if len(answer) > 300 else answer
        })

        print(f"INTENT: {q_repr.intent} | TYPE: {q_repr.question_type}")
        print(f"COVERAGE: {cov_state} ({cov_ratio*100:.0f}%) | Missing: {missing}")
        print(f"ANTI-COPY: {'PASS' if is_syn else 'FAIL'} (score={score:.2f}, reason: {reason})")
        print(f"CITATIONS ({len(citations)}): {citations}")
        print(f"EVIDENCE STRENGTH: {why.evidence_strength if why else 'N/A'}")
        print(f"\nFINAL PROCESSED ANSWER:\n{answer}\n")

    print("\n" + "="*80)
    print("SUMMARY RESULTS TABLE:")
    print("="*80)
    all_passed = True
    for r in results:
        status_str = "[PASS]" if r["passed"] else "[FAIL]"
        if not r["passed"]:
            all_passed = False
        print(f"Q{r['num']}: {status_str} Intent: {r['intent']:<20} | Type: {r['q_type']:<20} | Synth: {r['is_synthesized']} (score={r['overlap_score']:.2f}) | Cits: {r['citations']} | Strength: {r['evidence_strength']}")

    print("="*80)
    print(f"OVERALL STATUS: {'ALL 10/10 PASSED' if all_passed else 'SOME FAILED'}")
    print("="*80 + "\n")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
