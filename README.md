# Research Mind

Research Mind is an AI-powered scientific research assistant that uses Retrieval-Augmented Generation (RAG) to answer complex research questions grounded in scientific literature with verifiable citations and provenance.

---

## Current Development Phase: Phase 5 — Retrieval Evaluation & Refinement

> [!IMPORTANT]
> **Status Note**: We have completed **Phase 5: Quantitative Retrieval Evaluation & Production Refinement**. Dense vector search uses persistent local **ChromaDB** (`data/indexes/chroma/`), lexical search uses **BM25**, and candidate ranks are fused using **Reciprocal Rank Fusion (RRF)**. Based on Phase 5 ablation findings, production retrieval dynamically excludes non-evidence sections (`Header`, `References`, `Bibliography`, `Acknowledgements`).

---

## Architecture Overview

```
chunks.jsonl (10,147 semantic chunks across 250 research papers)
    ↓
BGE-small-en-v1.5 embeddings (13,718 384-dim normalized units)
    ↓
┌───────────────────────────────┬───────────────────────────────┐
│     Dense Vector Search       │     Lexical Keyword Search    │
│  ChromaDB (Cosine Distance)   │    BM25 (BM25Okapi, k1=1.5)   │
└───────────────────────────────┴───────────────────────────────┘
    ↓                                   ↓
Dense Ranks (top_k=20)              Lexical Ranks (top_k=20)
    └───────────────────────┬───────────┘
                            ↓
               Reciprocal Rank Fusion (RRF)
               RRF(d) = 1/(60 + r_dense) + 1/(60 + r_bm25)
                            ↓
               Retrieval-Time Section Filtering
               (Excludes Header, References, Bibliography, Acknowledgements)
                            ↓
               Parent Chunk Deduplication
                            ↓
               Top-10 Evidence Retrieval Results
```

---

## Dataset Domains & Corpus Target

Research Mind covers 5 canonical research domains with **250 valid research papers** (50 papers per domain):

1. **Artificial Intelligence (`artificial_intelligence`)** — ID Prefix: `AI` (50 papers)
2. **Cybersecurity (`cybersecurity`)** — ID Prefix: `CY` (50 papers)
3. **Agriculture (`agriculture`)** — ID Prefix: `AG` (50 papers)
4. **Healthcare (`healthcare`)** — ID Prefix: `HC` (50 papers)
5. **Climate (`climate`)** — ID Prefix: `CL` (50 papers)

Across the 250 papers, the corpus contains:
- **3,822** validated document sections (`data/processed/sections.jsonl`)
- **10,147** semantic chunks (`data/processed/chunks.jsonl`)
- **13,718** embedding units (including sub-split units for oversized chunks)

---

## Indexing & Dense Vector Store

- **Dense Vector Store:** Persistent ChromaDB database located at `data/indexes/chroma/` (Collection: `research_mind_chunks`, Cosine distance metric).
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (384 dimensions, L2-normalized).
- **Lexical Index:** BM25 (`BM25Okapi`) located at `data/indexes/bm25.pkl`.
- **Section Filtering:** Dynamic retrieval-time exclusion of non-evidence section types based on Phase 5 ablation findings (`Header`, `References`, `Bibliography`, `Acknowledgements`).
- **Provenance Retention:** Every retrieved result contains complete provenance metadata (`unit_id` $\rightarrow$ `parent_chunk_id` $\rightarrow$ `chunk_id` $\rightarrow$ `section_id` $\rightarrow$ `paper_id` $\rightarrow$ page range $\rightarrow$ domain/subtopic).

---

## Project Structure

```
researchmind/
├── config/
│   ├── domains.json             # Central configuration for 5 domains, ID prefixes & subtopics
│   ├── indexing.json            # Embedding model & ChromaDB indexing configuration
│   └── retrieval.json           # Baseline hybrid retrieval, RRF & section filtering configuration
├── data/
│   ├── papers/                  # Raw scientific PDFs (250 papers)
│   ├── metadata/
│   │   └── papers.json          # Master metadata repository for 250 corpus papers
│   ├── processed/
│   │   ├── sections.jsonl       # Extracted document sections (3,822 sections)
│   │   └── chunks.jsonl         # Token-bounded semantic chunks (10,147 chunks)
│   └── indexes/
│       ├── chroma/              # Persistent local ChromaDB database (13,718 records)
│       ├── bm25.pkl             # BM25Okapi lexical index
│       ├── chunk_embeddings.npy # Cached 13,718 vector embeddings
│       ├── faiss.index          # Reference/rollback FAISS vector index
│       └── faiss_mapping.json   # Reference/rollback FAISS mapping metadata
├── evaluation/
│   ├── test_queries.json        # 50 research evaluation queries across 5 domains
│   ├── ground_truth.json        # Precision-focused ground truth dataset (971 relevant mappings)
│   ├── results.json             # Structured evaluation metrics payload
│   └── error_analysis.md        # Categorized retrieval failure mode analysis
├── src/
│   ├── pipeline/
│   │   ├── chroma_indexer.py    # ChromaDB index manager & Cosine distance search
│   │   ├── indexing.py          # BM25 indexer & FAISS indexer
│   │   ├── embedding.py         # Sub-chunking & BGE vector embedding generator
│   │   ├── evaluation.py         # RetrievalEvaluator metrics suite (Hit, Precision, Recall, MRR, nDCG)
│   │   └── retrieval.py         # HybridRetriever service (ChromaDB + BM25 + RRF + Section Filtering)
├── scripts/
│   ├── build_chromadb.py        # Ingestion runner populating ChromaDB from embedding cache
│   ├── validate_chromadb.py     # 14 integrity validation checks for ChromaDB
│   ├── build_ground_truth.py    # Ground truth construction script
│   ├── run_evaluation.py        # 50-query evaluation pipeline runner
│   └── sanity_check_gt.py       # Ground truth manual sanity check runner
├── tests/                       # Unit test suite for chunking, indexing, ChromaDB, retrieval & evaluation
└── README.md
```

---

## Testing & Validation

Run all unit tests via `pytest`:

```bash
python -m pytest tests/test_evaluation.py tests/test_retrieval.py tests/test_indexing.py tests/test_chunking.py -v
```

Execute full 50-query retrieval evaluation:

```bash
python -m scripts.run_evaluation
```
