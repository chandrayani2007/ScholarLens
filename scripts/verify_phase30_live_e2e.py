"""
Phase 30 Live End-to-End Verification Script
Runs 4 live queries across in-corpus, out-of-corpus, and multi-domain queries
and verifies local retrieval, sufficiency decision, online fallback, citations, and Why This Answer metrics.
"""

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline.rag import RAGPipeline


def run_verification():
    pipeline = RAGPipeline()

    queries = [
        ("TEST 1 — In-Corpus Query", "What are the limitations of retrieval-augmented generation systems?"),
        ("TEST 2 — Out-of-Corpus Query 1", "What is quantum teleportation?"),
        ("TEST 3 — Out-of-Corpus Query 2", "What are recent advances in tokamak fusion energy confinement?"),
        ("TEST 4 — Multi-Domain Query", "How is AI used in medical image analysis?"),
    ]

    results = []

    for label, q in queries:
        print(f"\n========================================================")
        print(f"RUNNING {label}: '{q}'")
        print(f"========================================================")

        res = pipeline.answer(q)

        output = {
            "test_label": label,
            "query": q,
            "answer": res.answer,
            "citations": {k: v.to_dict() if hasattr(v, 'to_dict') else v for k, v in res.citations.items()},
            "why_this_answer": res.why_this_answer.to_dict(),
            "retrieval_metadata": res.retrieval_metadata,
            "domain_scope": res.domain_scope,
            "confidence": res.confidence,
        }

        print(f"Answer Preview:\n{res.answer[:300]}...\n")
        print(f"Source Type: {res.why_this_answer.source_type}")
        print(f"Confidence / Evidence Strength: {res.confidence}")
        print(f"Local Evidence Count: {res.retrieval_metadata.get('local_evidence_count')}")
        print(f"Online Evidence Count: {res.retrieval_metadata.get('online_evidence_count')}")
        print(f"Contributing Papers: {res.why_this_answer.contributing_papers}")
        print(f"Citations Used: {list(res.citations.keys())}")
        print(f"Bullet Points:\n" + "\n".join(res.why_this_answer.bullet_points))

        results.append(output)

    with open("data/phase30_live_verification_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nPhase 30 Live Verification Results successfully saved to data/phase30_live_verification_results.json")


if __name__ == "__main__":
    run_verification()
