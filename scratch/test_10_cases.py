import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

import logging
import json
from src.pipeline.llm import get_llm_provider
from src.pipeline.rag import RAGPipeline, QuestionRouter

logging.basicConfig(level=logging.INFO)

llm = get_llm_provider("gemini")
pipeline = RAGPipeline(llm_provider=llm)

test_cases = [
    {"num": 1, "scope": "AI038", "question": "What research problem does this paper address?", "expected_mode": "PAPER_MODE"},
    {"num": 2, "scope": "AI038", "question": "What methodology does this paper propose?", "expected_mode": "PAPER_MODE"},
    {"num": 3, "scope": "AI038", "question": "What are the main contributions of this paper?", "expected_mode": "PAPER_MODE"},
    {"num": 4, "scope": "AI038", "question": "What are the main results?", "expected_mode": "PAPER_MODE"},
    {"num": 5, "scope": "AI038", "question": "Give a comprehensive summary of this paper.", "expected_mode": "PAPER_MODE"},
    {"num": 6, "scope": "AI038", "question": "What is RAG?", "expected_mode": "GENERAL_MODE"},
    {"num": 7, "scope": "AI038", "question": "What is Agentic RAG?", "expected_mode": "GENERAL_MODE"},
    {"num": 8, "scope": "AI038", "question": "How does semantic search work?", "expected_mode": "GENERAL_MODE"},
    {"num": 9, "scope": None, "question": "What is RAG?", "expected_mode": "GENERAL_MODE"},
    {"num": 10, "scope": None, "question": "What is Agentic RAG?", "expected_mode": "GENERAL_MODE"},
]

results = []

for tc in test_cases:
    print(f"\n==================================================")
    print(f"TEST CASE {tc['num']}: Scope={tc['scope']} | Question='{tc['question']}'")
    print(f"==================================================")
    
    filters = {"paper_id": tc["scope"]} if tc["scope"] else None
    
    target_mode, route_cat = QuestionRouter.classify(
        question=tc["question"],
        paper_id_filter=tc["scope"]
    )
    
    print(f"Classified Mode: {target_mode} (Expected: {tc['expected_mode']}) | Route Category: {route_cat}")
    
    mode_pass = (target_mode == tc['expected_mode'])
    
    res = pipeline.answer(
        question=tc["question"],
        filters=filters,
        top_k=10,
        request_id=f"TC-{tc['num']}"
    )
    
    print(f"Returned Answer Length: {len(res.answer)}")
    print(f"Answer Preview:\n{res.answer[:250]}...")
    print(f"Citations: {list(res.citations.keys())}")
    print(f"Evidence Strength: {res.why_this_answer.evidence_strength}")
    print(f"Source Type: {res.why_this_answer.source_type}")
    
    results.append({
        "case": tc["num"],
        "question": tc["question"],
        "scope": tc["scope"],
        "expected_mode": tc["expected_mode"],
        "actual_mode": target_mode,
        "mode_pass": mode_pass,
        "citations": list(res.citations.keys()),
        "evidence_strength": res.why_this_answer.evidence_strength,
        "answer_preview": res.answer[:150]
    })

print("\n" + "="*80)
print("10 TEST CASES SUMMARY TABLE")
print("="*80)
print(f"{'#':<3} | {'Scope':<6} | {'Expected Mode':<14} | {'Actual Mode':<14} | {'Mode Pass':<9} | {'Cits Count':<10}")
print("-" * 75)
for r in results:
    print(f"{r['case']:<3} | {str(r['scope']):<6} | {r['expected_mode']:<14} | {r['actual_mode']:<14} | {str(r['mode_pass']):<9} | {len(r['citations']):<10}")
