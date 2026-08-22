"""
Corpus Indexing Runner (ChromaDB + BM25)

Supports:
- --pilot: 5-paper pilot run on AI001, CY001, AG001, HC001, CL001 with retrieval demonstration
- Full corpus indexing on all 10,147 chunks
- Oversized chunk (>512 tokens) sub-splitting
- BGE query instruction search demonstration
- Production artifact creation: ChromaDB persistent vector collection, bm25.pkl, index_metadata.json
"""

import json
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.pipeline.embedding import IndexingConfig, EmbeddingGenerator
from src.pipeline.indexing import BM25Indexer, write_index_metadata
from src.pipeline.chroma_indexer import ChromaDBIndexer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PILOT_PAPERS = {"AI001", "CY001", "AG001", "HC001", "CL001"}

PILOT_QUERIES = [
    {
        "domain": "Artificial Intelligence",
        "query": "What is retrieval augmented generation?",
    },
    {
        "domain": "Cybersecurity",
        "query": "How are network attacks detected?",
    },
    {
        "domain": "Agriculture",
        "query": "How is IoT used in smart farming?",
    },
    {
        "domain": "Healthcare",
        "query": "How does medical image analysis support diagnosis?",
    },
    {
        "domain": "Climate",
        "query": "How is machine learning used for climate prediction?",
    },
]


def format_text_preview(text: str, max_len: int = 150) -> str:
    clean = text.replace("\n", " ").strip()
    preview = clean[:max_len]
    return preview.encode("ascii", errors="replace").decode("ascii") + ("..." if len(clean) > max_len else "")


def run_indexing(is_pilot: bool = False):
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "config" / "indexing.json"
    CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"

    if is_pilot:
        INDEX_DIR = BASE_DIR / "data" / "indexes_pilot"
        CHROMA_DIR = INDEX_DIR / "chroma"
        BM25_PATH = INDEX_DIR / "bm25_pilot.pkl"
        META_PATH = INDEX_DIR / "index_metadata_pilot.json"
        print("=== PILOT MODE: Processing 5 representative papers ===")
        print(f"Pilot Papers: {sorted(PILOT_PAPERS)}")
    else:
        INDEX_DIR = BASE_DIR / "data" / "indexes"
        CHROMA_DIR = INDEX_DIR / "chroma"
        BM25_PATH = INDEX_DIR / "bm25.pkl"
        META_PATH = INDEX_DIR / "index_metadata.json"
        print("=== FULL MODE: Processing all 10,147 chunks ===")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load config
    config = IndexingConfig.from_file(CONFIG_PATH)
    print(f"\nIndexing Configuration:")
    print(f"  model: {config.embedding_model}")
    print(f"  dimension: {config.embedding_dim}")
    print(f"  max_seq_length: {config.max_seq_length}")
    print(f"  query_instruction: '{config.query_instruction}'")
    print(f"  sub_chunk_max_tokens: {config.sub_chunk_max_tokens}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  normalize_embeddings: {config.normalize_embeddings}")

    # 2. Load chunks
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"\nLoaded {len(chunks)} total chunks from {CHUNKS_PATH}")

    if is_pilot:
        chunks = [c for c in chunks if c["paper_id"] in PILOT_PAPERS]
        print(f"Filtered to {len(chunks)} chunks for 5 pilot papers")

    start_time = time.time()

    # 3. Sub-chunk processing & Embedding generation
    generator = EmbeddingGenerator(config)

    print("\nProcessing sub-chunks for oversized chunks (>512 tokens)...")
    units = generator.prepare_units(chunks)

    oversized_count = sum(1 for c in chunks if c.get("token_count", 0) > config.max_seq_length)
    print(f"  Original chunks: {len(chunks)}")
    print(f"  Oversized chunks (>512 tokens): {oversized_count}")
    print(f"  Total embedding units: {len(units)}")
    print(f"  Sub-units added: {len(units) - len(chunks)}")

    print(f"\nGenerating embeddings using {config.embedding_model}...")
    embed_start = time.time()
    embeddings = generator.encode_units(units, show_progress=True)
    embed_end = time.time()

    embed_duration = embed_end - embed_start
    avg_embed_per_unit = (embed_duration / len(units)) * 1000 if units else 0
    print(f"Generated {len(embeddings)} vectors in {embed_duration:.2f}s ({avg_embed_per_unit:.2f} ms/unit)")

    # Save vector cache
    generator.save_cache(embeddings, units, INDEX_DIR)

    # 4. Build ChromaDB Dense Vector Collection
    print("\nBuilding ChromaDB Dense Vector Collection...")
    chroma_start = time.time()
    chroma_indexer = ChromaDBIndexer(db_dir=CHROMA_DIR, collection_name="research_mind_chunks", dim=config.embedding_dim)
    chroma_indexer.build_index(embeddings, units)
    chroma_end = time.time()
    print(f"ChromaDB collection built in {chroma_end - chroma_start:.2f}s")

    # 5. Build BM25 index
    print("\nBuilding BM25Okapi Index...")
    bm25_start = time.time()
    bm25_indexer = BM25Indexer(k1=config.bm25_k1, b=config.bm25_b)
    bm25_indexer.build_index(units)
    bm25_indexer.save(BM25_PATH)
    bm25_end = time.time()
    print(f"BM25 index built and saved in {bm25_end - bm25_start:.2f}s")

    total_duration = time.time() - start_time

    # 6. Save Metadata
    extra_stats = {
        "total_execution_time_seconds": round(total_duration, 2),
        "embedding_time_seconds": round(embed_duration, 2),
        "avg_embed_ms_per_unit": round(avg_embed_per_unit, 2),
        "chroma_build_time_seconds": round(chroma_end - chroma_start, 2),
        "bm25_build_time_seconds": round(bm25_end - bm25_start, 2),
        "artifact_sizes_bytes": {
            "bm25_pickle": BM25_PATH.stat().st_size if BM25_PATH.exists() else 0,
        },
    }

    write_index_metadata(
        META_PATH,
        config,
        total_original_chunks=len(chunks),
        total_embedding_units=len(units),
        dense_vector_count=chroma_indexer.collection.count(),
        bm25_doc_count=len(bm25_indexer.units),
        extra=extra_stats,
    )

    # 7. If Pilot: Run 5 Representative Queries
    if is_pilot:
        print(f"\n{'='*70}")
        print("PILOT RETRIEVAL DEMONSTRATION & RELEVANCE INSPECTION")
        print(f"{'='*70}")

        for item in PILOT_QUERIES:
            domain = item["domain"]
            query_str = item["query"]

            print(f"\n----------------------------------------------------------------------")
            print(f"DOMAIN: {domain}")
            print(f"QUERY:  \"{query_str}\"")
            print(f"----------------------------------------------------------------------")

            query_vec = generator.encode_queries([query_str])

            chroma_results = chroma_indexer.search(query_vec[0], k=3)
            bm25_results = bm25_indexer.search(query_str, k=3)

            print("\n  [ChromaDB Dense Top-3 Results]:")
            for rank, r in enumerate(chroma_results, 1):
                preview = format_text_preview(r["text"], max_len=120)
                print(f"    #{rank} [Score: {r['score']:.4f}] unit_id: {r['unit_id']} (paper: {r['paper_id']})")
                print(f"       section: {r['section_name']} | pages: {r['page_start']}-{r['page_end']}")
                print(f"       text: {preview}")

            print("\n  [BM25 Lexical Top-3 Results]:")
            for rank, r in enumerate(bm25_results, 1):
                preview = format_text_preview(r["text"], max_len=120)
                print(f"    #{rank} [Score: {r['score']:.4f}] unit_id: {r['unit_id']} (paper: {r['paper_id']})")
                print(f"       section: {r['section_name']} | pages: {r['page_start']}-{r['page_end']}")
                print(f"       text: {preview}")

        print(f"\n{'='*70}")
        print("PILOT COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")


def main():
    is_pilot = "--pilot" in sys.argv
    run_indexing(is_pilot=is_pilot)


if __name__ == "__main__":
    main()
