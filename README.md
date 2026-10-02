# ScholarLens (Research Mind)

**ScholarLens** is an explainable, multi-domain AI scientific research assistant powered by Retrieval-Augmented Generation (RAG). It provides evidence-grounded answers to complex research questions across 1,000 scientific papers with claim-level verification, neural cross-encoder reranking, provenance tracking, and an embedded PDF reader interface.

---

> [!IMPORTANT]
> **System Architecture & Operational Status**  
> ScholarLens features an end-to-end multi-domain RAG framework:
> - **Corpus Size**: 1,000 peer-reviewed papers (200 papers across 5 domains: Artificial Intelligence, Cybersecurity, Agriculture, Healthcare, Climate).
> - **Hybrid Retrieval**: Dense Vector Search (ChromaDB + `BAAI/bge-small-en-v1.5`) combined with Lexical Keyword Search (`BM25Okapi`) fused via Reciprocal Rank Fusion (RRF).
> - **Neural Cross-Encoder Reranking**: Re-scores candidate passages using `BAAI/bge-reranker-base` full token-level cross-attention.
> - **Claim Grounding Validator**: Filters unsupported assertions and aligns inline citations (`[E1]`, `[O1]`, `[U1]`) to evidence passages.
> - **Online Fallback**: Automatic arXiv academic fallback when local corpus evidence is insufficient.
> - **In-App PDF Viewer**: Full inline PDF viewer modal supporting local corpus PDFs, uploaded custom documents, and online academic papers.

---

## 🏗️ System Architecture

```
                       User Question / Query (Web Interface or API)
                                           │
                                           ▼
                      Question Decomposition & Intent Analysis
                    (Concept Extraction, Expected Section Mapping)
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
           Dense Vector Search                        Lexical Keyword Search
     ChromaDB (Cosine Distance, BGE)                BM25 (BM25Okapi, k1=1.5)
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                              Reciprocal Rank Fusion (RRF)
                         RRF(d) = 1/(60 + r_dense) + 1/(60 + r_bm25)
                                           │
                                           ▼
                           Retrieval-Time Section Filtering
                 (Excludes Header, References, Bibliography, Acknowledgements)
                                           │
                                           ▼
                          Neural Cross-Encoder Rescoring
                            (BAAI/bge-reranker-base)
                                           │
                                           ▼
                          Evidence Sufficiency Judgment
                     ┌─────────────────────┴─────────────────────┐
            [Sufficient]                                    [Insufficient]
                 │                                                │
                 ▼                                                ▼
     Local Evidence Assembly                        arXiv Online Fallback Search
                 │                                                │
                 └─────────────────────┬──────────────────────────┘
                                       ▼
                       LLM Answer Synthesis & Generation
                                       │
                                       ▼
                         Claim Grounding Validation
              (Strips Unsupported Claims & Aligns Citations)
                                       │
                                       ▼
              React + Vite Glassmorphic UI with Embedded PDF Viewer
```

---

## 📚 Dataset Domains & Corpus Metrics

ScholarLens indexes 1,000 scientific papers across 5 canonical research domains (200 papers per domain):

1. **Artificial Intelligence (`artificial_intelligence`)** — ID Prefix: `AI` (200 papers, `AI001`–`AI200`)
2. **Cybersecurity (`cybersecurity`)** — ID Prefix: `CY` (200 papers, `CY001`–`CY200`)
3. **Agriculture (`agriculture`)** — ID Prefix: `AG` (200 papers, `AG001`–`AG200`)
4. **Healthcare (`healthcare`)** — ID Prefix: `HC` (200 papers, `HC001`–`HC200`)
5. **Climate (`climate`)** — ID Prefix: `CL` (200 papers, `CL001`–`CL200`)

---

## ✨ Key Features

- 🔍 **Hybrid Retrieval Engine**: Fuses dense semantic vector embeddings (`BAAI/bge-small-en-v1.5`) with BM25 lexical keyword matching via Reciprocal Rank Fusion (RRF).
- 🧠 **Neural Cross-Encoder Reranking**: Uses `BAAI/bge-reranker-base` to evaluate full query-passage cross-attention scores.
- 🛡️ **Claim Grounding & Verification**: Inspects synthesized answers against retrieved passages and strips unevidenced or hallucinated claims.
- 🌐 **Online Academic Fallback**: Queries live arXiv search when local evidence is insufficient for specialized topics.
- 📄 **Embedded PDF Reader Modal**: Allows users to read genuine PDF documents for local corpus papers, custom uploaded files, or online sources directly in the web app.
- 📁 **Custom PDF Ingestion**: Extract, chunk, and ask questions directly over user-uploaded PDF papers.
- 📊 **Provenance & Transparency**: Every claim links to explicit citation tags (`[E1]`, `[O1]`, `[U1]`) mapped to paper titles, section names, and page ranges.

---

## 📁 Project Structure

```
researchmind/
├── app/                         # FastAPI Web Application & API Server
│   ├── main.py                  # Server entry point & CORS configuration
│   ├── database.py              # SQLite database session & ORM configuration
│   ├── models/                  # User, SavedQuery, and History ORM models
│   └── routers/
│       ├── auth.py              # User authentication & JWT token generation
│       ├── corpus.py            # Corpus browsing, search, metadata & PDF streaming endpoint
│       └── research.py          # Research query pipeline endpoint & custom PDF upload router
├── config/
│   ├── domains.json             # 5 domain definitions, prefixes & subtopics
│   ├── indexing.json            # Embedding model & ChromaDB configuration
│   └── retrieval.json           # Baseline RRF hybrid retrieval & section filtering configuration
├── data/
│   ├── papers/                  # 1,000 PDF scientific research papers across 5 domain folders
│   ├── metadata/
│   │   └── papers.json          # Master metadata repository for 1,000 corpus papers
│   ├── processed/
│   │   ├── sections.jsonl       # Extracted document sections
│   │   └── chunks.jsonl         # Token-bounded semantic chunks
│   └── indexes/
│       ├── chroma/              # Persistent ChromaDB vector database
│       └── bm25.pkl             # BM25Okapi lexical index
├── docs/
│   └── ScholarLens_IEEE_Paper.tex # IEEE Conference Paper Manuscript for ScholarLens
├── frontend/                    # Modern React + Vite Web Application
│   ├── src/
│   │   ├── components/          # AnswerCard, EvidenceAccordion, KeySourcesPanel, PdfViewerModal, Header
│   │   ├── pages/               # ResearchPage, CorpusPage, SavedQueriesPage, AuthPages
│   │   └── services/            # Centralized API client (`api.js`)
│   └── vite.config.js           # Vite dev server configuration & API proxy rules
├── evaluation/
│   ├── test_queries.json        # Research evaluation queries across 5 domains
│   ├── ground_truth.json        # Ground truth relevance mapping dataset
│   ├── results.json             # Quantitative evaluation results
│   └── error_analysis.md        # Categorized retrieval error analysis
├── src/
│   └── pipeline/
│       ├── chroma_indexer.py    # ChromaDB manager (Cosine distance)
│       ├── chunking.py          # Sentence-boundary semantic chunking
│       ├── embedding.py         # Sub-chunking & BGE embedding generator
│       ├── evaluation.py        # Metrics suite (Hit, Precision, Recall, MRR, nDCG)
│       ├── llm.py               # LLM abstraction layer & MockLLMProvider generator
│       ├── online_fallback.py   # arXiv online academic search fallback
│       ├── pdf_extractor.py     # PDF text extraction & structure analyzer
│       ├── rag.py               # RAG Pipeline orchestrator & ClaimGroundingValidator
│       └── retrieval.py         # HybridRetriever (ChromaDB + BM25 + RRF + Section Filter)
├── scripts/
│   ├── build_chromadb.py        # Populates ChromaDB index
│   ├── run_evaluation.py        # Evaluates 50-query retrieval benchmark
│   └── validate_chromadb.py     # Runs integrity checks on ChromaDB index
├── tests/                       # Complete pytest unit test suite (58+ tests)
└── README.md
```

---

## ⚡ Getting Started

### Prerequisites

- Python 3.10+ (Python 3.11/3.13 recommended)
- Node.js 18+ and `npm`

### 1. Install Backend Dependencies

```bash
# Install Python requirements
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## 🚀 Running ScholarLens Locally

### Option A: Start FastAPI Backend API Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API server will run at `http://127.0.0.1:8000`. API documentation is available at `http://127.0.0.1:8000/docs`.

### Option B: Start React + Vite Frontend Dev Server

In a second terminal:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser to interact with the ScholarLens Web App.

---

## 🧪 Testing & Evaluation

### Run Unit Tests

Execute unit tests covering chunking, indexing, retrieval, claim grounding, and answer quality:

```bash
python -m pytest tests/test_answer_quality.py tests/test_answer_quality_enhancements.py tests/test_retrieval.py tests/test_indexing.py tests/test_chunking.py -v
```

### Run Retrieval Evaluation Benchmark

Execute quantitative 50-query evaluation pipeline across Hit Rate, MRR, Precision, Recall, and nDCG:

```bash
python -m scripts.run_evaluation
```
