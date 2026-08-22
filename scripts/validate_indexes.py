"""
Phase 3 & 4 — Index Validation Script

Comprehensive integrity checks for ChromaDB, BM25, vector embeddings, and metadata.
Supports --pilot flag for pilot validation.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from src.pipeline.chroma_indexer import ChromaDBIndexer


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    is_pilot = "--pilot" in sys.argv

    if is_pilot:
        INDEX_DIR = BASE_DIR / "data" / "indexes_pilot"
        CHROMA_DIR = INDEX_DIR / "chroma"
        BM25_PATH = INDEX_DIR / "bm25_pilot.pkl"
        META_PATH = INDEX_DIR / "index_metadata_pilot.json"
        CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks_pilot.jsonl"
        if not CHUNKS_PATH.exists():
            CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
        print(f"Validating PILOT Indexes in {INDEX_DIR}")
    else:
        INDEX_DIR = BASE_DIR / "data" / "indexes"
        CHROMA_DIR = INDEX_DIR / "chroma"
        BM25_PATH = INDEX_DIR / "bm25.pkl"
        META_PATH = INDEX_DIR / "index_metadata.json"
        CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
        print(f"Validating FULL Production Indexes in {INDEX_DIR}")

    NPY_PATH = INDEX_DIR / "chunk_embeddings.npy"

    # Load chunks
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    if is_pilot:
        pilot_papers = {"AI001", "CY001", "AG001", "HC001", "CL001"}
        chunks = [c for c in chunks if c["paper_id"] in pilot_papers]
    chunk_ids = {c["chunk_id"] for c in chunks}

    # Load metadata
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load ChromaDB collection
    chroma_indexer = ChromaDBIndexer(db_dir=CHROMA_DIR, collection_name="research_mind_chunks", dim=metadata["embedding_config"]["embedding_dim"])
    chroma_count = chroma_indexer.collection.count()

    # Load BM25 index
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)
    bm25_units = bm25_data["units"]

    # Load vector cache
    embeddings = np.load(NPY_PATH)

    results = []
    all_pass = True

    def check(name, condition, detail=""):
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        results.append((name, status, detail))
        icon = "[OK]" if condition else "[!!]"
        print(f"  {icon} [{status}] {name}" + (f" -- {detail}" if detail else ""))

    print("\n=== Index Integrity Checks ===")

    # 1. ChromaDB record count matches embedding array rows
    check(
        "ChromaDB vector count matches embedding matrix rows",
        chroma_count == embeddings.shape[0],
        f"{chroma_count} ChromaDB records / {embeddings.shape[0]} matrix rows"
    )

    # 2. Vector dimension == 384
    expected_dim = metadata["embedding_config"]["embedding_dim"]
    check(
        f"Embedding dimension matches config ({expected_dim})",
        embeddings.shape[1] == expected_dim,
        f"matrix d={embeddings.shape[1]}"
    )

    # 3. No NaN or Inf vectors
    has_nan_inf = np.isnan(embeddings).any() or np.isinf(embeddings).any()
    check("No NaN or Inf values in embeddings", not has_nan_inf, "0 NaNs/Infs found")

    # 4. L2 Normalization check (norm approx 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    norm_ok = np.allclose(norms, 1.0, atol=1e-3)
    check("Vectors L2 normalized (norm approx 1.0)", norm_ok, f"norm range: {norms.min():.4f} - {norms.max():.4f}")

    # 5. BM25 doc count matches ChromaDB count
    check(
        "BM25 document count matches ChromaDB count",
        len(bm25_units) == chroma_count,
        f"{len(bm25_units)} BM25 docs / {chroma_count} ChromaDB records"
    )

    # 6. Parent chunk linkage integrity
    invalid_parents = []
    for item in bm25_units:
        parent_id = item["parent_chunk_id"]
        if parent_id not in chunk_ids:
            invalid_parents.append((item["unit_id"], parent_id))
    check(
        "Parent chunk IDs match original corpus chunks",
        len(invalid_parents) == 0,
        f"{len(invalid_parents)} invalid parent linkages" if invalid_parents else "all 100% valid"
    )

    # 7. Complete provenance fields present in every BM25 unit
    required_fields = ["unit_id", "parent_chunk_id", "paper_id", "section_id", "section_name", "page_start", "page_end", "domain", "subtopic", "text"]
    missing_fields = []
    for item in bm25_units:
        for field in required_fields:
            if field not in item:
                missing_fields.append((item.get("unit_id", ""), field))
    check(
        "Provenance fields completeness in BM25 corpus",
        len(missing_fields) == 0,
        f"{len(missing_fields)} missing fields" if missing_fields else "all present"
    )

    # 8. All expected papers represented
    indexed_papers = {item["paper_id"] for item in bm25_units}
    if is_pilot:
        expected_papers = {"AI001", "CY001", "AG001", "HC001", "CL001"}
    else:
        expected_papers = {c["paper_id"] for c in chunks}
    missing_papers = expected_papers - indexed_papers
    check(
        "Paper coverage",
        len(missing_papers) == 0,
        f"{len(indexed_papers)}/{len(expected_papers)} papers" + (f", missing: {sorted(missing_papers)}" if missing_papers else "")
    )

    # 9. All 5 domains represented
    domains = {item["domain"] for item in bm25_units}
    expected_domains = {"agriculture", "artificial_intelligence", "climate", "cybersecurity", "healthcare"}
    check(
        "All 5 domains represented in index",
        expected_domains.issubset(domains),
        f"domains present: {sorted(domains)}"
    )

    print(f"\n{'='*50}")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    total = len(results)
    print(f"RESULT: {passed}/{total} checks passed")
    if all_pass:
        print("ALL INDEX CHECKS PASSED")
    else:
        print("SOME INDEX CHECKS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
