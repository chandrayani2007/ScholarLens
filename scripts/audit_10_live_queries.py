"""
Live Audit Script for the 10 User-Specified Scientific Queries
Evaluates:
1. Question Intent & Deconstructed Concepts
2. Local Evidence Sufficiency & Generic Answerability Decision
3. Decision: Local Answer vs. Online Academic Fallback vs. Insufficient Evidence
4. Source Attribution ([E#] / [O#]), Citations, and Why This Answer Verification
"""

import sys
import json
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.rag import RAGPipeline, EvidenceSufficiencyEvaluator

TEST_QUERIES = [
    "What algorithms are commonly used for intrusion detection?",
    "What are the limitations of retrieval-augmented generation systems?",
    "How does Explainable AI improve trust in healthcare?",
    "What is quantum teleportation?",
    "How does reinforcement learning improve robotic manipulation?",
    "What datasets are commonly used for intrusion detection?",
    "What are the major challenges in climate prediction?",
    "How does CRISPR-Cas9 genome editing work?",
    "What is BM25 and how does it rank documents?",
    "What are recent advances in fusion energy?"
]

def run_audit():
    print("=" * 80)
    print("STARTING LIVE AUDIT OF 10 SCIENTIFIC QUERIES")
    print("=" * 80)
    
    pipeline = RAGPipeline()
    audit_results = []

    for idx, q in enumerate(TEST_QUERIES, start=1):
        print(f"\n[{idx}/10] Query: '{q}'")
        decomp = EvidenceSufficiencyEvaluator.decompose_query(q)
        print(f"       Intent: {decomp['intent']} | Relational: {decomp['is_relational']}")
        print(f"       Content Words: {decomp['all_content_words']}")

        res = pipeline.answer(q)

        local_cits = [k for k in res.citations.keys() if k.startswith("E")]
        online_cits = [k for k in res.citations.keys() if k.startswith("O")]
        contributing = res.why_this_answer.contributing_papers
        source_type = res.why_this_answer.source_type
        strength = res.why_this_answer.evidence_strength

        print(f"       Source Type: {source_type}")
        print(f"       Evidence Strength: {strength}")
        print(f"       Local Citations ({len(local_cits)}): {local_cits}")
        print(f"       Online Citations ({len(online_cits)}): {online_cits}")
        print(f"       Contributing Papers: {contributing}")
        print(f"       Answer Excerpt: {res.answer[:160]}...")

        audit_results.append({
            "index": idx,
            "query": q,
            "intent": decomp["intent"],
            "is_relational": decomp["is_relational"],
            "source_type": source_type,
            "evidence_strength": strength,
            "local_citations_count": len(local_cits),
            "online_citations_count": len(online_cits),
            "contributing_papers": contributing,
            "answer_preview": res.answer[:250],
            "bullet_points": res.why_this_answer.bullet_points
        })

    with open("live_10_queries_audit_result.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE - RESULTS SAVED TO live_10_queries_audit_result.json")
    print("=" * 80)

if __name__ == "__main__":
    run_audit()
