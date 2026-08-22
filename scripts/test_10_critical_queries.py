"""
Test Suite & Manual Inspection Runner for 10 Critical User Queries

Enforces the central invariant:
TOPICAL RELEVANCE ≠ QUESTION ANSWERABILITY

Validates for 10 target queries:
1. Question decomposition & QuestionContract extraction
2. Primary & secondary local evidence evaluation
3. Online academic fallback trigger condition
4. Final answer generation OR honest insufficient evidence response
5. Why This Answer card fields & Evidence Strength
"""

import sys
import json
import logging
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rag import RAGPipeline, GenericQuestionAnalyzer, GenericEvidenceEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Test10Queries")

TEST_QUERIES = [
    "What is cloud computing?",
    "What are the limitations of LLMs?",
    "What algorithms are commonly used for intrusion detection?",
    "How does reinforcement learning improve robotic manipulation?",
    "What are the benefits of precision agriculture?",
    "What causes model drift in machine learning?",
    "How does carbon capture work?",
    "What is quantum teleportation?",
    "What are the major challenges in climate prediction?",
    "How is AI used in medical image analysis?",
]


def run_10_queries_audit():
    pipeline = RAGPipeline()
    results = []

    print("=" * 80)
    print("RUNNING 10 CRITICAL QUERIES AUDIT")
    print("=" * 80)

    for idx, q in enumerate(TEST_QUERIES, 1):
        print(f"\n[{idx}/10] Query: '{q}'")
        q_repr = GenericQuestionAnalyzer.analyze(q)

        # Execute RAG Pipeline
        response = pipeline.answer(q)

        meta = response.retrieval_metadata
        why = response.why_this_answer

        audit_entry = {
            "query_index": idx,
            "question": q,
            "contract": q_repr.contract.to_dict() if q_repr.contract else {},
            "intent": q_repr.intent,
            "requested_aspect": q_repr.requested_aspect,
            "local_eval_decision": meta.get("local_eval_decision", "N/A"),
            "concept_support": meta.get("concept_support", 0.0),
            "intent_support": meta.get("intent_support", 0.0),
            "relationship_support": meta.get("relationship_support", 0.0),
            "answerability_score": meta.get("sufficiency_score", 0.0),
            "online_fallback_active": meta.get("online_fallback_active", False),
            "used_citations_count": meta.get("used_citations_count", 0),
            "evidence_strength": why.evidence_strength,
            "explanation_summary": why.explanation_summary,
            "answer_preview": response.answer[:250] + "..." if len(response.answer) > 250 else response.answer,
        }

        results.append(audit_entry)

        print(f"  • Intent: {audit_entry['intent']}")
        print(f"  • Concept: {audit_entry['contract'].get('concept')}")
        print(f"  • Local Eval Decision: {audit_entry['local_eval_decision']}")
        print(f"  • Concept Support: {audit_entry['concept_support']:.2f}")
        print(f"  • Intent Support: {audit_entry['intent_support']:.2f}")
        print(f"  • Answerability Score: {audit_entry['answerability_score']:.2f}")
        print(f"  • Online Fallback Triggered: {audit_entry['online_fallback_active']}")
        print(f"  • Evidence Strength: {audit_entry['evidence_strength']}")
        print(f"  • Explanation Summary: {audit_entry['explanation_summary']}")
        print(f"  • Answer Snippet: {audit_entry['answer_preview'][:150]}")
        print("-" * 80)

    # Save detailed JSON output for manual inspection
    output_file = PROJECT_ROOT / "scratch" / "audit_10_critical_queries.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nAudit complete! Detailed results written to: {output_file}")


if __name__ == "__main__":
    run_10_queries_audit()
