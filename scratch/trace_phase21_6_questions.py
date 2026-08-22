"""
Scratch script to trace the 6 required Phase 21 benchmark questions through RAGPipeline
and print complete evidence text, claims, citations, Key Sources, and Why This Answer summaries.
"""

import json
from src.pipeline.rag import RAGPipeline
from src.pipeline.llm import MockLLMProvider

pipeline = RAGPipeline(llm_provider=MockLLMProvider())

questions = [
    ("What is retrieval augmented generation?", {"domain": "artificial_intelligence"}),
    ("How can IoT improve smart irrigation?", {"domain": "agriculture"}),
    ("How can deep learning detect cyber attacks?", {"domain": "cybersecurity"}),
    ("How is deep learning used in medical image diagnosis?", {"domain": "healthcare"}),
    ("How can machine learning help predict climate patterns?", {"domain": "climate"}),
    ("What is photosynthesis?", {"domain": "agriculture"}),
]

for idx, (q, filters) in enumerate(questions, 1):
    print(f"\n==================================================")
    print(f"QUESTION {idx}: {q} (filters={filters})")
    print(f"==================================================")
    res = pipeline.answer(q, filters=filters)
    
    print("\n--- GENERATED ANSWER ---")
    print(res.answer)
    
    print("\n--- FINAL CITATIONS & KEY SOURCES ---")
    for cit_id, cit in res.citations.items():
        print(f"[{cit_id}] -> Paper {cit.paper_id} | Section: {cit.section_name} | Pages: {cit.pages}")
        
    print("\n--- EVIDENCE STRENGTH & WHY THIS ANSWER ---")
    print(f"Strength: {res.why_this_answer.evidence_strength}")
    print(f"Summary: {res.why_this_answer.explanation_summary}")
    
    print("\n--- RETRIEVED E1..E10 CANDIDATE PASSAGES ---")
    for ev in res.evidence[:3]:
        print(f"[{ev.citation_id}] Paper {ev.paper_id} ({ev.domain}) - '{ev.section_name}':")
        print(f"    {ev.text[:140]}...")
