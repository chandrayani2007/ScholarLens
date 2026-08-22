"""
Live Citation & Claim Grounding Verification Audit Script

Audits 8 mandatory queries across AI, Cybersecurity, Agriculture, Quantum Computing, Climate, etc.:
1. What are the limitations of Agentic RAG?
2. What is cloud computing?
3. What algorithms are commonly used for intrusion detection?
4. What are the limitations of LLMs?
5. How does reinforcement learning improve robotic manipulation?
6. What are the benefits of precision agriculture?
7. What is quantum teleportation?
8. How does carbon capture work?

For each query, inspects:
- final answer text
- factual claims & citation alignment
- exact cited passages
- answerability decision (LOCAL_SUFFICIENT vs ONLINE_SUFFICIENT vs INSUFFICIENT)
- Evidence Strength & Why This Answer metrics
- local vs online source counts
"""

import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("live_citation_audit")

from src.pipeline.rag import RAGPipeline

AUDIT_QUERIES = [
    "What are the limitations of agentic RAG?",
    "What are the limitations of large language models?",
    "What algorithms are commonly used for intrusion detection?",
    "How does federated learning preserve privacy?",
    "What are the main challenges of deploying deep learning models on edge devices?",
    "How does explainable AI improve trust in healthcare?",
    "What are the benefits of smart irrigation systems?",
    "What causes climate prediction uncertainty?",
    "How does CRISPR-Cas9 work?",
    "What is quantum teleportation?",
    "What are the advantages of retrieval-augmented generation?",
    "How does reinforcement learning improve robotic manipulation?",
    "Compare cloud computing and edge computing.",
    "What are the main components of a transformer architecture?",
    "What are recent advances in multimodal AI?",
]

def run_audit():
    logger.info("Initializing RAGPipeline...")
    pipeline = RAGPipeline()
    results = []

    for i, q in enumerate(AUDIT_QUERIES, 1):
        print(f"\n==================================================")
        print(f"QUERY {i}/{len(AUDIT_QUERIES)}: {q}")
        print(f"==================================================")

        res = pipeline.answer(q)

        local_cits = [k for k in res.citations.keys() if k.startswith("E")]
        online_cits = [k for k in res.citations.keys() if k.startswith("O")]

        audit_record = {
            "query": q,
            "answer_preview": res.answer[:250] + "..." if len(res.answer) > 250 else res.answer,
            "evidence_strength": res.why_this_answer.evidence_strength,
            "unsupported_claims": res.why_this_answer.unsupported_claims,
            "contributing_papers": res.why_this_answer.contributing_papers,
            "active_citations": list(res.citations.keys()),
            "local_citation_count": len(local_cits),
            "online_citation_count": len(online_cits),
            "source_type": res.why_this_answer.source_type,
            "local_eval_decision": res.retrieval_metadata.get("local_eval_decision", "N/A"),
            "online_fallback_active": res.retrieval_metadata.get("online_fallback_active", False),
        }

        print(f"Decision: {audit_record['local_eval_decision']} | Online Fallback: {audit_record['online_fallback_active']}")
        print(f"Source Type: {audit_record['source_type']}")
        safe_preview = audit_record['answer_preview'].encode('ascii', errors='replace').decode('ascii')
        print(f"Evidence Strength: {audit_record['evidence_strength']}")
        print(f"Active Citations: {audit_record['active_citations']}")
        print(f"Contributing Papers ({len(audit_record['contributing_papers'])}): {audit_record['contributing_papers']}")
        print(f"Unsupported Claims: {audit_record['unsupported_claims']}")
        print(f"\nAnswer Preview:\n{safe_preview}")

        results.append(audit_record)

    with open("live_10_queries_audit_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n==================================================")
    print(f"AUDIT COMPLETE. Summary saved to live_10_queries_audit_result.json.")
    print(f"==================================================")

if __name__ == "__main__":
    run_audit()
