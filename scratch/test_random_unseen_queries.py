"""
Script to test ScholarLens with brand-new, unseen, random research queries.
Tests:
1. Novel Corpus Query (AI / Transformer Positional Encoding)
2. Novel Online Academic Search Query (Quantum Neural Architecture Search)
3. Novel Single-Paper Mode Query (Custom paper uploaded context)
4. Novel Comparative Query (Graph Neural Networks vs Transformers)
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rag import RAGPipeline
from src.pipeline.retrieval import HybridRetriever
from src.pipeline.llm import get_llm_provider

def run_test():
    print("=" * 80)
    print("INITIALIZING SCHOLARLENS RAG PIPELINE FOR NOVEL UNSEEN QUERIES TEST")
    print("=" * 80)

    # Initialize retriever & pipeline
    retriever = HybridRetriever()
    llm_provider = get_llm_provider("gemini" if os.environ.get("GEMINI_API_KEY") else "mock")
    pipeline = RAGPipeline(retriever=retriever, llm_provider=llm_provider)

    test_queries = [
        {
            "name": "Test 1: Unseen Corpus Query (Transformer Positional Encoding)",
            "question": "How do transformer models utilize positional encoding to maintain word order without recurrence?",
            "filters": {"domain": "artificial_intelligence"},
        },
        {
            "name": "Test 2: Unseen Online Fallback Query (Quantum Neural Architecture Search)",
            "question": "What is Quantum Neural Architecture Search and how does it optimize quantum circuits?",
            "filters": None,
        },
        {
            "name": "Test 3: Unseen Single-Paper Mode Query (Uploaded Paper Context)",
            "question": "What is the primary technical advantage of dual-encoder prompting for low-resource languages?",
            "uploaded_paper_name": "dual_encoder_prompting_2026.pdf",
            "uploaded_paper_text": (
                "Dual-Encoder Prompting for Zero-Shot Cross-Lingual Transfer.\n"
                "Abstract: Low-resource cross-lingual transfer often suffers from catastrophic representation alignment drift when fine-tuning dense encoder models. "
                "In this paper, we introduce Dual-Encoder Prompting (DEP), a novel framework that decouples language-agnostic semantic prompts from language-specific acoustic/textual representations. "
                "Our empirical evaluations across 42 low-resource languages demonstrate that DEP achieves a 14.8% relative gain in zero-shot F1 score over standard monolingual fine-tuning. "
                "The primary technical advantage of Dual-Encoder Prompting is that it preserves source language feature geometry while enabling zero-shot inference without requiring target language parallel supervision."
            ),
        },
        {
            "name": "Test 4: Unseen Comparative Query (GNNs vs Transformers for Molecules)",
            "question": "How does graph neural network node classification compare to transformer self-attention for molecular property prediction?",
            "filters": None,
        },
    ]

    results = []

    for test in test_queries:
        print("\n" + "=" * 80)
        print(f"RUNNING {test['name']}")
        print(f"Question: {test['question']}")
        print("=" * 80)

        response = pipeline.answer(
            question=test['question'],
            filters=test.get('filters'),
            uploaded_paper_text=test.get('uploaded_paper_text'),
            uploaded_paper_name=test.get('uploaded_paper_name'),
            top_k=5,
        )

        res_dict = response.to_dict()
        results.append({
            "test_name": test["name"],
            "question": test["question"],
            "answer": response.answer,
            "confidence": response.confidence,
            "citations": list(response.citations.keys()),
            "evidence_count": len(response.evidence),
            "evidence_sources": [e.source_type for e in response.evidence],
            "why_this_answer_summary": response.why_this_answer.explanation_summary,
            "bullets": response.why_this_answer.bullet_points,
        })

        print(f"\n[CONFIDENCE]: {response.confidence}")
        print(f"\n[ANSWER]:\n{response.answer}")
        print(f"\n[CITATIONS]: {list(response.citations.keys())}")
        print(f"\n[WHY THIS ANSWER]: {response.why_this_answer.explanation_summary}")
        for b in response.why_this_answer.bullet_points:
            print(f"  {b}")

    # Save outputs to scratch
    output_path = Path(__file__).parent / "random_unseen_queries_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"ALL UNSEEN TEST QUERIES COMPLETED. Results saved to {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_test()
