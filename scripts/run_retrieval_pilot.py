"""
Phase 4 — Hybrid Retrieval 5-Domain Pilot Benchmark Script (ChromaDB + BM25 + RRF)

Executes sanity check queries across AI, Cybersecurity, Agriculture, Healthcare, and Climate.
Displays ChromaDB, BM25, and RRF fused top results with complete provenance metadata.
"""

import sys
import logging
from pathlib import Path

from src.pipeline.retrieval import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_QUERIES = [
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


def format_preview(text: str, max_len: int = 120) -> str:
    clean = text.replace("\n", " ").strip()
    preview = clean[:max_len]
    return preview.encode("ascii", errors="replace").decode("ascii") + ("..." if len(clean) > max_len else "")


def main():
    print("=" * 75)
    print("CHROMADB + BM25 HYBRID RETRIEVAL 5-DOMAIN PILOT BENCHMARK")
    print("=" * 75)
    print("Note: These five domain queries are sanity checks only to verify pipeline")
    print("integrity, RRF rank fusion, provenance mapping, and metadata filtering.")
    print("Actual retrieval quality evaluation will occur in later evaluation phases.")
    print("=" * 75)

    retriever = HybridRetriever()

    for item in BENCHMARK_QUERIES:
        domain = item["domain"]
        query_str = item["query"]

        print(f"\n" + "-" * 75)
        print(f"DOMAIN: {domain}")
        print(f"QUERY:  \"{query_str}\"")
        print("-" * 75)

        # Execute hybrid retrieval (RRF fused top-5 results)
        fused_results = retriever.retrieve(query_str, top_k=5)

        # Retrieve raw ChromaDB top-3 candidates for comparison
        query_vec = retriever.generator.encode_queries([query_str])
        chroma_raw = retriever.chroma_indexer.search(query_vec[0], k=3)

        # Retrieve raw BM25 top-3 candidates for comparison
        bm25_raw = retriever.bm25_indexer.search(query_str, k=3)

        print("\n  [ChromaDB Dense Candidates Top-3]:")
        for rank, r in enumerate(chroma_raw, 1):
            preview = format_preview(r["text"])
            print(f"    #{rank} [Similarity (1-dist): {r['score']:.4f}] unit_id: {r['unit_id']} (paper: {r['paper_id']})")
            print(f"       section: {r['section_name']} | pages: {r['page_start']}-{r['page_end']}")
            print(f"       text: {preview}")

        print("\n  [BM25 Lexical Candidates Top-3]:")
        for rank, r in enumerate(bm25_raw, 1):
            preview = format_preview(r["text"])
            print(f"    #{rank} [Score: {r['score']:.4f}] unit_id: {r['unit_id']} (paper: {r['paper_id']})")
            print(f"       section: {r['section_name']} | pages: {r['page_start']}-{r['page_end']}")
            print(f"       text: {preview}")

        print("\n  [RRF Fused Top-5 Final Results]:")
        for res in fused_results:
            preview = format_preview(res.text)
            methods_str = "+".join(res.retrieval_methods)
            scores_str = f"RRF: {res.rrf_score:.6f}"
            if res.dense_score is not None:
                scores_str += f" | ChromaSim: {res.dense_score:.4f}"
            if res.bm25_score is not None:
                scores_str += f" | BM25: {res.bm25_score:.4f}"

            print(f"    #{res.rank} [{scores_str}] Method: [{methods_str}]")
            print(f"       unit_id: {res.unit_id} (parent: {res.parent_chunk_id}, paper: {res.paper_id})")
            print(f"       section: {res.section_name} | pages: {res.page_start}-{res.page_end}")
            print(f"       text: {preview}")

    # Demonstrate metadata filtering sanity check
    print(f"\n" + "=" * 75)
    print("METADATA FILTERING SANITY CHECK (domain='healthcare')")
    print("=" * 75)
    filtered_results = retriever.retrieve(
        "What is machine learning?",
        top_k=3,
        filters={"domain": "healthcare"}
    )
    for res in filtered_results:
        preview = format_preview(res.text)
        print(f"  #{res.rank} [{res.domain}] paper: {res.paper_id} | section: {res.section_name}")
        print(f"     text: {preview}")

    print("\n" + "=" * 75)
    print("RETRIEVAL PILOT BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
