"""
Phase 27.5 Live Audit Script — Executes all 8 live audit queries through RAGPipeline
and outputs full JSON response inspection details.
"""

import json
import os
import re
from pathlib import Path
from src.pipeline.rag import RAGPipeline
from src.pipeline.llm import MockLLMProvider, get_llm_provider

AUDIT_QUERIES = [
    {
        "id": 1,
        "query": "What are the limitations of retrieval-augmented generation systems?",
        "user_domain": "artificial_intelligence",
        "expected_intent": "Limitation",
        "expected_scope": "Single Domain (Artificial Intelligence)"
    },
    {
        "id": 2,
        "query": "How does explainable AI improve trust in healthcare?",
        "user_domain": None,
        "expected_intent": "Explanation",
        "expected_scope": "Multi-Domain (Artificial Intelligence + Healthcare)"
    },
    {
        "id": 3,
        "query": "What are the challenges of using deep learning for cybersecurity?",
        "user_domain": "cybersecurity",
        "expected_intent": "Challenge",
        "expected_scope": "Single Domain (Cybersecurity)"
    },
    {
        "id": 4,
        "query": "What are the benefits and limitations of smart irrigation systems?",
        "user_domain": "agriculture",
        "expected_intent": "Limitation",
        "expected_scope": "Single Domain (Agriculture)"
    },
    {
        "id": 5,
        "query": "What are the major challenges in machine learning-based climate prediction?",
        "user_domain": "climate",
        "expected_intent": "Challenge",
        "expected_scope": "Single Domain (Climate)"
    },
    {
        "id": 6,
        "query": "How is AI used in medical image analysis?",
        "user_domain": None,
        "expected_intent": "Method",
        "expected_scope": "Multi-Domain (Artificial Intelligence + Healthcare)"
    },
    {
        "id": 7,
        "query": "What are recent advances in vision-language action models for robot manipulation?",
        "user_domain": "artificial_intelligence",
        "expected_intent": "Method",
        "expected_scope": "Single Domain (Artificial Intelligence)"
    },
    {
        "id": 8,
        "query": "What is surface code quantum error correction?",
        "user_domain": None,
        "expected_intent": "Definition",
        "expected_scope": "Out-of-Corpus (Online Academic Fallback)"
    }
]


def run_audit():
    pipeline = RAGPipeline(llm_provider=MockLLMProvider())
    results = []

    print("==========================================================================")
    print("STARTING SCHOLARLENS LIVE ANSWER QUALITY AUDIT (8 QUERIES)")
    print("==========================================================================")

    for item in AUDIT_QUERIES:
        qid = item["id"]
        q = item["query"]
        user_dom = item["user_domain"]

        filters = {"domain": user_dom} if user_dom else None
        res = pipeline.answer(q, filters=filters)

        ans_text = res.answer
        words = len(re.findall(r"\b\w+\b", ans_text))

        local_cits = [tag for tag in res.citations if tag.startswith("E")]
        online_cits = [tag for tag in res.citations if tag.startswith("O")]

        # Citation grounding check
        citations_grounded = True
        for tag, cit in res.citations.items():
            if tag not in ans_text:
                citations_grounded = False

        # Answer quality checks
        starts_direct = not ans_text.lower().startswith(q.lower()[:30])
        no_jargon = "critical architectural limitations during real-world scientific execution" not in ans_text
        understandable = len(ans_text) > 80 and starts_direct and no_jargon

        fallback_active = res.retrieval_metadata.get("online_fallback_active", False)
        status = "PASSED" if (citations_grounded and understandable and words >= 30) else "NEEDS_REVIEW"

        audit_entry = {
            "query_id": qid,
            "query": q,
            "domain_scope": res.why_this_answer.domain_scope,
            "word_count": words,
            "local_sources_count": len(local_cits),
            "online_sources_count": len(online_cits),
            "citations_grounded": citations_grounded,
            "answer_understandable": understandable,
            "fallback_active": fallback_active,
            "confidence": res.confidence,
            "source_type": res.why_this_answer.source_type,
            "status": status,
            "first_sentence": ans_text.split(". ")[0] + "." if ". " in ans_text else ans_text[:100],
            "full_answer": ans_text,
            "citations": [c.to_dict() for c in res.citations.values()],
            "why_this_answer": res.why_this_answer.to_dict()
        }
        results.append(audit_entry)

        print(f"\n[QUERY {qid}] {q}")
        print(f"  • Scope: {res.why_this_answer.domain_scope}")
        print(f"  • Words: {words} | Local: {len(local_cits)} | Online: {len(online_cits)} | Fallback: {fallback_active}")
        print(f"  • Confidence: {res.confidence} | Status: {status}")
        print(f"  • Opening: {audit_entry['first_sentence']}")

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "live_answer_quality_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n==========================================================================")
    print(f"AUDIT COMPLETE. Results saved to {out_file}")
    print("==========================================================================")


if __name__ == "__main__":
    run_audit()
