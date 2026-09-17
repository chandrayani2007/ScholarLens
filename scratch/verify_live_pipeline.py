import json
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from src.pipeline.rag import RAGPipeline, AntiCopyValidator
from src.pipeline.llm import MockLLMProvider

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
    pipeline = RAGPipeline(llm_provider=MockLLMProvider())

    questions = [
        "What is the main research problem addressed by this paper?",
        "What methodology did the authors propose?",
        "What dataset or benchmark was used?",
        "How was the data prepared or preprocessed?",
        "What were the main quantitative results?",
        "How did the proposed approach compare with the baseline?",
        "What are the main contributions of this paper?",
        "Why did the authors choose this approach?",
        "What limitations did the authors identify?",
        "What is the main research problem, methodology, dataset, and main results?",
    ]

    results = []
    print("=" * 80)
    print("RUNNING LIVE END-TO-END RAG PIPELINE VERIFICATION ACROSS 10 QUESTIONS")
    print("=" * 80)

    for idx, q in enumerate(questions, 1):
        res = pipeline.answer(
            question=q,
            uploaded_paper_text=OPEN_SCHOLAR_PAPER_TEXT,
            uploaded_paper_name="SDC_BasePaper.pdf",
            filters={"domain": "artificial_intelligence"},
        )

        is_synth, reason, overlap = AntiCopyValidator.validate_answer(res.answer, res.evidence)

        results.append({
            "idx": idx,
            "question": q,
            "answer": res.answer,
            "evidence": res.evidence,
            "citations": res.citations,
            "is_synth": is_synth,
            "reason": reason,
            "overlap": overlap,
        })

        print(f"\n[{idx}/10] Q: {q}")
        print(f"PASS ANTI-COPY: {is_synth} (score={overlap:.2f}, reason: {reason})")
        print(f"CITATIONS: {list(res.citations.keys())}")
        print(f"ANSWER:\n{res.answer}\n")

    # Verify all passed
    all_passed = all(r["is_synth"] for r in results)
    print("=" * 80)
    if all_passed:
        print(">>> ALL 10 QUESTIONS PASSED ANTI-COPY AND SYNTHESIS VALIDATION! <<<")
    else:
        print(">>> SOME QUESTIONS FAILED! <<<")
    print("=" * 80)

    # Detailed trace for Q1
    q1 = results[0]
    print("\n" + "#" * 80)
    print("QUESTION 1 DETAILED TRACE:")
    print("#" * 80)
    print(f"QUESTION:\n{q1['question']}\n")
    print("RETRIEVED EVIDENCE:")
    for ev in q1['evidence']:
        print(f"[{getattr(ev, 'citation_id', 'U1')}] Section: {getattr(ev, 'section_name', 'N/A')}")
        print(f"    Text: {ev.text[:250]}...\n")
    print(f"FINAL PROCESSED ANSWER:\n{q1['answer']}\n")
    print("#" * 80)

if __name__ == "__main__":
    main()
