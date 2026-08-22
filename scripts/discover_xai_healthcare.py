import json
import time
from src.connectors.arxiv import ArxivConnector

arxiv = ArxivConnector()

queries = [
    "explainable clinical AI",
    "explainable medical AI",
    "XAI medical imaging",
    "interpretable clinical decision",
    "explainable radiology AI",
    "explainable healthcare AI",
    "XAI healthcare",
    "interpretable medical diagnosis",
]

print("=== Healthcare-Specific XAI Candidate Discovery ===\n")

all_candidates = []
seen_ids = set()

for q in queries:
    print(f"Query: '{q}'")
    candidates = arxiv.search_candidates(query=q, domain="healthcare", subtopic="explainable_ai_in_healthcare", max_results=10)
    new_count = 0
    for c in candidates:
        if c.source_id not in seen_ids:
            seen_ids.add(c.source_id)
            all_candidates.append((q, c))
            new_count += 1
            print(f"  [{len(all_candidates):2d}] {c.title}")
            print(f"       arXiv: {c.source_id} | Year: {c.publication_year}")
    print(f"  -> {len(candidates)} results, {new_count} new unique\n")
    time.sleep(4)

print(f"\nTotal unique candidates: {len(all_candidates)}")
