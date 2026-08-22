"""
Phase 28.6 — Step 1 & 2: Diagnostic Analysis of Retrieval Failures and Chunk Quality

Analyzes failure cases in evaluation/results/ to categorize root causes:
A. Relevant text missing from paper extraction
B. Relevant information exists but chunking is poor
C. Query vocabulary differs from paper vocabulary
D. BM25 lexical mismatch
E. Dense embedding mismatch
F. RRF weighting problem
G. Section metadata problem
H. Domain filtering problem
I. Query intent / wrapper word problem
J. Paper itself is weakly relevant to query
"""

import csv
import json
import re
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(r"c:\Users\nugur\Desktop\researchmind")
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
PAPERS_JSON_PATH = BASE_DIR / "data" / "metadata" / "papers.json"

def diagnose():
    coverage_csv = RESULTS_DIR / "1000_paper_coverage.csv"
    realistic_csv = RESULTS_DIR / "realistic_query_results.csv"
    failures_csv = RESULTS_DIR / "retrieval_failures.csv"

    print("==========================================================================")
    print("=== PHASE 28.6 STEP 1: RETRIEVAL FAILURE DIAGNOSIS ===")
    print("==========================================================================")

    # 1. Analyze 1,000-paper coverage results
    unretrieved_papers = []
    if coverage_csv.exists():
        with open(coverage_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["retrieved"] == "False" or row["rank"] == "None":
                    unretrieved_papers.append(row)

    print(f"Total Papers Unretrieved in Top-10 (out of 1000): {len(unretrieved_papers)} / 1000 ({len(unretrieved_papers)/10.0:.1f}%)")

    # Group unretrieved papers by domain
    unretrieved_by_dom = defaultdict(int)
    for p in unretrieved_papers:
        unretrieved_by_dom[p["domain"]] += 1

    for dom, cnt in sorted(unretrieved_by_dom.items()):
        print(f"  - {dom}: {cnt} unretrieved papers ({cnt/200.0*100:.1f}%)")

    # 2. Analyze Realistic Queries Failures
    realistic_failures = []
    if failures_csv.exists():
        with open(failures_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                realistic_failures.append(row)

    print(f"\nTotal Realistic Query Failures Logged: {len(realistic_failures)} / 290")

    # Categorize causes based on query intent & keywords
    cause_breakdown = defaultdict(int)
    classified_failures = []

    for fail in realistic_failures:
        q_id = fail["query_id"]
        q_text = fail["question"]
        cat = fail["category"]

        # Determine specific cause
        q_lower = q_text.lower()
        if "online_fallback" in cat:
            cause = "Out-of-Corpus Query (Expected Online Fallback)"
            code = "K. Paper itself absent in corpus"
        elif any(w in q_lower for w in ["function in research", "evaluated in", "addressed in", "significance of"]):
            cause = "Query Intent Wrapper Word Mismatch in Relevance Gate"
            code = "I. Query intent wrapper word overhead"
        elif any(w in q_lower for w in ["lora", "dino", "bge", "resnet", "yolo", "bert", "gpt", "bioner"]):
            cause = "Technical Acronym / Specific Vocabulary Mismatch"
            code = "C. Query vocabulary differs from paper vocabulary"
        else:
            cause = "Abstract Text Chunking Density Constraint"
            code = "B. Information chunking density"

        cause_breakdown[code] += 1
        classified_failures.append({
            "query_id": q_id,
            "question": q_text,
            "category": cat,
            "cause_code": code,
            "description": cause
        })

    print("\nRoot Cause Categorization Breakdown:")
    for code, cnt in sorted(cause_breakdown.items()):
        print(f"  [{code}]: {cnt} queries ({cnt/float(len(realistic_failures))*100:.1f}%)")

    # Export retrieval_failure_analysis.csv
    analysis_csv = RESULTS_DIR / "retrieval_failure_analysis.csv"
    with open(analysis_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "question", "category", "cause_code", "description"])
        writer.writeheader()
        writer.writerows(classified_failures)

    print(f"\nSaved diagnosis to {analysis_csv}")
    print("==========================================================================")

if __name__ == "__main__":
    diagnose()
