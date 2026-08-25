import json
import logging
import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline.rag import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

QUESTIONS = [
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

def main():
    print("=" * 80)
    print("SCHOLARLENS LIVE 15-BENCHMARK-QUERY PIPELINE AUDIT")
    print("=" * 80)

    pipeline = RAGPipeline()
    audit_results = []

    for idx, q in enumerate(QUESTIONS, 1):
        print(f"\n[{idx}/15] QUERY: '{q}'")
        try:
            resp = pipeline.answer(q, filters={"domain": "all"})
            
            meta = resp.retrieval_metadata
            local_decision = meta.get("local_eval_decision", "N/A")
            online_active = meta.get("online_fallback_active", False)
            local_cnt = meta.get("local_evidence_count", 0)
            online_cnt = meta.get("online_evidence_count", 0)
            cits = list(resp.citations.keys())
            papers = resp.why_this_answer.contributing_papers
            strength = resp.why_this_answer.evidence_strength

            item = {
                "id": idx,
                "question": q,
                "intent": meta.get("intent", "Unknown"),
                "local_decision": local_decision,
                "local_evidence_count": local_cnt,
                "online_fallback_active": online_active,
                "online_evidence_count": online_cnt,
                "evidence_strength": strength,
                "confidence": resp.confidence,
                "citations": cits,
                "contributing_papers": papers,
                "unsupported_claims": resp.why_this_answer.unsupported_claims,
                "answer_snippet": resp.answer[:200].replace("\n", " ") + "...",
                "why_this_answer": resp.why_this_answer.explanation_summary[:150],
            }
            audit_results.append(item)

            print(f"  [-] Intent: {item['intent']}")
            print(f"  [-] Local Decision: {local_decision}")
            print(f"  [-] Online Fallback Triggered: {online_active}")
            print(f"  [-] Evidence Count: Local={local_cnt}, Online={online_cnt}")
            print(f"  [-] Strength: {strength} | Citations: {cits}")
            print(f"  [-] Papers: {papers}")
            print(f"  [-] Answer Snippet: {item['answer_snippet']}")

        except Exception as e:
            print(f"  [-] ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results to json
    out_file = BASE_DIR / "scratch" / "live_15_queries_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\nAudit complete! Saved results to {out_file}")

if __name__ == "__main__":
    main()
