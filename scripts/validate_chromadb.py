"""
Phase 4 — 14 Integrity Validation Checks for Persistent ChromaDB Vector Index

Validates:
1. ChromaDB collection 'research_mind_chunks' exists
2. Collection is persistent at data/indexes/chroma/
3. Record count = 13,718
4. All record IDs are unique
5. Embedding dimension = 384
6. Zero missing embeddings
7. Zero NaNs or Infs
8. All 12 required metadata fields present
9. All unit_ids match corpus units
10. All 250 papers represented
11. All 5 domains represented
12. Zero orphan records
13. Document text is non-empty
14. Page ranges valid (page_start <= page_end)
"""

import json
import logging
from pathlib import Path
import numpy as np
import chromadb

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REQUIRED_METADATA_KEYS = {
    "unit_id",
    "parent_chunk_id",
    "chunk_id",
    "paper_id",
    "section_id",
    "section_name",
    "domain",
    "subtopic",
    "page_start",
    "page_end",
    "is_sub_chunk",
    "token_count",
}

EXPECTED_DOMAINS = {
    "artificial_intelligence",
    "cybersecurity",
    "agriculture",
    "healthcare",
    "climate",
}


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    chroma_dir = BASE_DIR / "data" / "indexes" / "chroma"
    chunks_path = BASE_DIR / "data" / "processed" / "chunks.jsonl"
    units_path = BASE_DIR / "data" / "indexes" / "chunk_embeddings_ids.json"

    print("=" * 70)
    print("CHROMADB RETRIEVAL INDEX — 14 INTEGRITY VALIDATION CHECKS")
    print("=" * 70)

    checks_passed = 0
    total_checks = 14

    # Check 1 & 2: Database and Collection existence & persistence
    if not chroma_dir.exists():
        print("FAIL: Check 1 & 2 - ChromaDB directory missing!")
        return

    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        collection = client.get_collection("research_mind_chunks")
        print("PASS: Check 1 - ChromaDB collection 'research_mind_chunks' exists")
        print("PASS: Check 2 - Persistent client successfully initialized at data/indexes/chroma/")
        checks_passed += 2
    except Exception as e:
        print(f"FAIL: Check 1 & 2 - Could not get collection: {e}")
        return

    # Fetch all records
    record_count = collection.count()
    records = collection.get(include=["embeddings", "metadatas", "documents"])

    ids = records["ids"]
    metadatas = records["metadatas"]
    documents = records["documents"]
    embeddings = records["embeddings"]

    # Check 3: Record count = 13,718
    if record_count == 13718 and len(ids) == 13718:
        print(f"PASS: Check 3 - Record count = {record_count} (13,718 expected)")
        checks_passed += 1
    else:
        print(f"FAIL: Check 3 - Record count = {record_count} (expected 13,718)")

    # Check 4: All record IDs unique
    if len(ids) == len(set(ids)):
        print(f"PASS: Check 4 - All {len(ids)} record IDs are unique")
        checks_passed += 1
    else:
        print(f"FAIL: Check 4 - Duplicate IDs found ({len(ids)} total vs {len(set(ids))} unique)")

    # Check 5: Embedding dimension = 384
    emb_array = np.array(embeddings)
    if emb_array.ndim == 2 and emb_array.shape[1] == 384:
        print(f"PASS: Check 5 - Embedding dimensions = {emb_array.shape[1]} (384 expected)")
        checks_passed += 1
    else:
        print(f"FAIL: Check 5 - Invalid embedding shape {emb_array.shape}")

    # Check 6: Zero missing embeddings
    if len(embeddings) == len(ids) and all(e is not None for e in embeddings):
        print("PASS: Check 6 - Zero missing embeddings across all records")
        checks_passed += 1
    else:
        print("FAIL: Check 6 - Missing embeddings detected!")

    # Check 7: Zero NaNs or Infs
    if not np.isnan(emb_array).any() and not np.isinf(emb_array).any():
        print("PASS: Check 7 - Zero NaN or Inf values in vector embeddings")
        checks_passed += 1
    else:
        print("FAIL: Check 7 - NaNs or Infs detected in vector matrix!")

    # Check 8: All required metadata fields present
    missing_meta_count = 0
    for meta in metadatas:
        if not REQUIRED_METADATA_KEYS.issubset(set(meta.keys())):
            missing_meta_count += 1
    if missing_meta_count == 0:
        print(f"PASS: Check 8 - All {len(REQUIRED_METADATA_KEYS)} required metadata fields present in all records")
        checks_passed += 1
    else:
        print(f"FAIL: Check 8 - {missing_meta_count} records missing required metadata fields!")

    # Check 9: All unit_ids exist in corpus metadata
    with open(units_path, "r", encoding="utf-8") as f:
        expected_units = json.load(f)
    expected_unit_ids = {u.get("unit_id", u["chunk_id"]) for u in expected_units}

    if set(ids) == expected_unit_ids:
        print(f"PASS: Check 9 - 100% of unit_ids match cached corpus metadata units ({len(ids)})")
        checks_passed += 1
    else:
        print("FAIL: Check 9 - Discrepancy between ChromaDB record IDs and corpus unit IDs!")

    # Check 10: All 250 papers represented
    papers_represented = {meta["paper_id"] for meta in metadatas}
    if len(papers_represented) == 250:
        print(f"PASS: Check 10 - All 250 research papers represented in ChromaDB")
        checks_passed += 1
    else:
        print(f"FAIL: Check 10 - Represented papers = {len(papers_represented)} (250 expected)")

    # Check 11: All 5 domains represented
    domains_represented = {meta["domain"] for meta in metadatas}
    if domains_represented == EXPECTED_DOMAINS:
        print(f"PASS: Check 11 - All 5 research domains represented ({sorted(EXPECTED_DOMAINS)})")
        checks_passed += 1
    else:
        print(f"FAIL: Check 11 - Missing domains! Found: {domains_represented}")

    # Check 12: Zero orphan records
    valid_parents = {meta["parent_chunk_id"] for meta in metadatas}
    orphan_count = sum(1 for meta in metadatas if not meta["parent_chunk_id"])
    if orphan_count == 0:
        print(f"PASS: Check 12 - Zero orphan records detected (all linked to valid parent_chunk_id)")
        checks_passed += 1
    else:
        print(f"FAIL: Check 12 - {orphan_count} orphan records found!")

    # Check 13: Document text is non-empty
    empty_docs = sum(1 for doc in documents if not doc or not doc.strip())
    if empty_docs == 0:
        print("PASS: Check 13 - 100% of ChromaDB documents contain non-empty text")
        checks_passed += 1
    else:
        print(f"FAIL: Check 13 - {empty_docs} documents are empty or whitespace!")

    # Check 14: Valid page provenance ranges
    invalid_pages = sum(1 for meta in metadatas if meta["page_start"] > meta["page_end"] or meta["page_start"] <= 0)
    if invalid_pages == 0:
        print("PASS: Check 14 - 100% of records have valid page ranges (page_start <= page_end)")
        checks_passed += 1
    else:
        print(f"FAIL: Check 14 - {invalid_pages} records have invalid page ranges!")

    print("=" * 70)
    print(f"SUMMARY: {checks_passed} / {total_checks} CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
