# Phase 5 — Refined Retrieval Error Analysis

**Generated At:** 2026-08-14  
**Evaluation Dataset:** 50 Queries across 5 Domains (AI, Cybersecurity, Agriculture, Healthcare, Climate)  
**Evaluated Engine:** Production Hybrid RRF with Retrieval-Time Section Filtering  
**Ground-Truth Dataset:** Refined Precision-Focused Ground Truth (Average 19.4 units/query)

---

## 1. Summary of System Performance & Error Distribution

With retrieval-time section filtering enabled (`Header`, `References`, `Bibliography`, `Acknowledgements` excluded), **0 non-evidence headers or citation references appear in top retrieved positions**.

| Performance Metric | Count / Score | Description |
| :--- | :---: | :--- |
| **Top 1 Exact Match (Hit@1)** | **64.0% (32 / 50)** | Query core answered in rank #1 passage |
| **Top 5 Coverage (Hit@5)** | **78.0% (39 / 50)** | Relevant evidence retrieved in Top 5 candidates |
| **Top 10 Coverage (Hit@10)** | **88.0% (44 / 50)** | Relevant evidence retrieved in Top 10 candidates |
| **Sub-optimal Rank (Hit@1=0, Hit@10>0)** | **12 Queries** | Relevant evidence present in Top 10 but ranked below position #1 |
| **Zero Relevant Retrieved (Hit@10=0)** | **6 Queries** | Highly specific query terms with low overall corpus density |

---

## 2. Analysis of Remaining Failure Modes

### 2.1 Highly Specific Queries with Low Corpus Density
- **Example Queries:** `CY_Q002` (zero-day exploit defense), `HC_Q006` (CRISPR off-target therapeutics), `CL_Q004` (tropical cyclone intensity).
- **Cause:** These specific sub-topics are mentioned in only a small handful of corpus papers. While general domain context exists, exact precision-matched chunks are sparse.

### 2.2 Semantic vs Lexical Mismatch
- **Cause:** BM25 term frequency scores generic introductory background passages containing repeated query words higher than dense semantic embedding passages describing specific methods.

---

## 3. Section Filtering Impact Verification

Section filtering successfully eliminated 100% of header noise and reference list false positives, producing an **+8.0% gain in Hit@1**, an **+8.0% gain in Hit@5**, and an **+11.8% gain in nDCG@10**.
