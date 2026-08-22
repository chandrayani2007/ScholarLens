"""
Phase 28.5 — Step 1: Corpus Index Coverage Verification Script

Verifies:
1. All 1,000 papers in data/metadata/papers.json exist.
2. Chunk counts per paper in data/processed/chunks.jsonl.
3. ChromaDB collection 'research_mind_chunks' contains embeddings for all 1,000 papers.
4. BM25 index contains all 1,000 papers.
5. Identifies any missing papers or zero-chunk papers.
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
import chromadb

BASE_DIR = Path(r"c:\Users\nugur\Desktop\researchmind")
PAPERS_JSON_PATH = BASE_DIR / "data" / "metadata" / "papers.json"
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"
CHROMA_DIR = BASE_DIR / "data" / "indexes" / "chroma"
BM25_PATH = BASE_DIR / "data" / "indexes" / "bm25.pkl"


def verify_index_coverage():
    with open(PAPERS_JSON_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)

    all_paper_ids = {p["paper_id"] for p in papers}
    paper_domain_map = {p["paper_id"]: p["domain"] for p in papers}
    total_papers = len(papers)

    print("==========================================================================")
    print("=== STEP 1: RESEARCH MIND 1,000-PAPER INDEX COVERAGE VERIFICATION ===")
    print("==========================================================================")
    print(f"Total papers in metadata: {total_papers}")

    # 1. Inspect chunks.jsonl
    chunks_per_paper = defaultdict(int)
    total_chunks = 0
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                pid = chunk["paper_id"]
                chunks_per_paper[pid] += 1
                total_chunks += 1

    chunks_paper_ids = set(chunks_per_paper.keys())

    # 2. Inspect ChromaDB Collection
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection("research_mind_chunks")
    chroma_count = collection.count()

    # Get distinct paper_ids from ChromaDB metadata
    # Fetch metadata in batches to get distinct paper_ids
    chroma_paper_ids = set()
    offset = 0
    batch_size = 2000
    while offset < chroma_count:
        res = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
        for meta in res["metadatas"]:
            if meta and "paper_id" in meta:
                chroma_paper_ids.add(meta["paper_id"])
        offset += batch_size

    # 3. Inspect BM25 Index
    with open(BM25_PATH, "rb") as f:
        bm25_data = pickle.load(f)

    bm25_units = bm25_data.get("units", [])
    bm25_paper_ids = {u["paper_id"] for u in bm25_units if "paper_id" in u}

    # 4. Compute Coverage & Discrepancies
    missing_in_chunks = all_paper_ids - chunks_paper_ids
    missing_in_chroma = all_paper_ids - chroma_paper_ids
    missing_in_bm25 = all_paper_ids - bm25_paper_ids
    zero_chunk_papers = {pid for pid in all_paper_ids if chunks_per_paper[pid] == 0}

    indexed_in_all = chunks_paper_ids.intersection(chroma_paper_ids).intersection(bm25_paper_ids)
    missing_overall = all_paper_ids - indexed_in_all

    print(f"\nTotal structural chunks: {total_chunks}")
    print(f"Total ChromaDB vector units: {chroma_count}")
    print(f"Total BM25 index units: {len(bm25_units)}")
    print(f"\nIndexed Papers:")
    print(f"  In chunks.jsonl: {len(chunks_paper_ids)} / 1000")
    print(f"  In ChromaDB:    {len(chroma_paper_ids)} / 1000")
    print(f"  In BM25 index:   {len(bm25_paper_ids)} / 1000")
    print(f"  In ALL indexes:  {len(indexed_in_all)} / 1000")

    print(f"\nCoverage Summary:")
    print(f"  Total papers:            {total_papers}")
    print(f"  Indexed papers:          {len(indexed_in_all)}")
    print(f"  Missing papers:          {len(missing_overall)}")
    print(f"  Papers with zero chunks: {len(zero_chunk_papers)}")

    if missing_overall:
        print(f"\nMissing Paper IDs: {sorted(list(missing_overall))}")

    print("--------------------------------------------------------------------------")
    if len(indexed_in_all) == 1000 and len(missing_overall) == 0 and len(zero_chunk_papers) == 0:
        print("[OK] 1,000 / 1,000 PAPERS PHYSICALLY INDEXED AND VERIFIED!")
    else:
        print("[FAIL] INDEX COVERAGE DISCREPANCY DETECTED!")
    print("--------------------------------------------------------------------------")

    return {
        "total_papers": total_papers,
        "indexed_papers": len(indexed_in_all),
        "missing_papers": len(missing_overall),
        "zero_chunk_papers": len(zero_chunk_papers),
        "total_chunks": total_chunks,
        "total_vectors": chroma_count
    }


if __name__ == "__main__":
    verify_index_coverage()
