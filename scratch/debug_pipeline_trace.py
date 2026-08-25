"""
End-to-End Pipeline Debugger for ScholarLens
Logs complete step-by-step pipeline execution for candidate queries.
"""

import os
import sys
import json
import logging

sys.path.insert(0, os.path.abspath("."))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline.rag import RAGPipeline, GenericQuestionAnalyzer, GenericEvidenceEvaluator, DomainScopeDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TEST_QUESTIONS = [
    "What are the limitations of Agentic RAG?",
    "What algorithms are commonly used for intrusion detection?",
    "What are the limitations of large language models?",
    "How does federated learning preserve privacy?",
    "What are the main challenges of deploying deep learning models on edge devices?",
]

def safe_str(s: str) -> str:
    return str(s).encode('ascii', 'replace').decode('ascii')

def debug_pipeline():
    pipeline = RAGPipeline()
    print("=" * 80)
    print("PIPELINE DEBUG TRACE FOR USER'S 5 TEST QUESTIONS")
    print("=" * 80)

    for idx, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{idx}] QUESTION: '{q}'")
        q_repr = GenericQuestionAnalyzer.analyze(q)
        scope_res = DomainScopeDetector.detect(q)
        print(f"  Intent: {q_repr.intent}")
        print(f"  Requested Aspect: {q_repr.requested_aspect}")
        print(f"  Main Subject: {q_repr.main_subject}")
        print(f"  Detected Scope: {scope_res.scope_type.value} | Allowed Domains: {scope_res.allowed_domains}")

        # Local retrieval
        retrieved_results = pipeline.retriever.retrieve(
            q, top_k=10, allowed_domains=scope_res.allowed_domains, intent=q_repr.intent
        )
        print(f"  Local Retrieved Chunks: {len(retrieved_results)}")
        if retrieved_results:
            for r_idx, r in enumerate(retrieved_results[:3], 1):
                p_text = safe_str(getattr(r, 'text', '')[:120].replace('\n', ' '))
                p_paper = getattr(r, 'paper_id', '')
                p_domain = getattr(r, 'domain', '')
                print(f"    Passage {r_idx}: [{p_paper}] domain={p_domain} text='{p_text}...'")

        local_eval = GenericEvidenceEvaluator.evaluate(q, q_repr, retrieved_results, "LOCAL")
        print(f"  Local Eval: related={local_eval.related} answerable={local_eval.answerable} decision={local_eval.decision} score={local_eval.answerability_score:.2f} direct_passages={len(local_eval.direct_supporting_passages)}")
        print(f"  Missing Aspects: {local_eval.missing_aspects}")
        print(f"  Rationale: {safe_str(local_eval.rationale[:150])}")

        # Full pipeline answer call
        res = pipeline.answer(q)
        print(f"  Final Source Tag: {res.why_this_answer.source_type}")
        print(f"  Final Confidence: {res.confidence}")
        print(f"  Final Citations: {list(res.citations.keys())}")
        print(f"  Final Answer Snippet:\n{safe_str(res.answer[:300])}")
        print("-" * 80)

if __name__ == "__main__":
    debug_pipeline()
