"""
Phase 4 — Populate Persistent ChromaDB Collection from Existing Embeddings Cache

Reuses:
- data/indexes/chunk_embeddings.npy (13,718 x 384 vectors)
- data/indexes/chunk_embeddings_ids.json (13,718 metadata units)

Builds:
- data/indexes/chroma/ (ChromaDB PersistentClient collection 'research_mind_chunks')
"""

import json
import logging
from pathlib import Path
import numpy as np

from src.pipeline.chroma_indexer import ChromaDBIndexer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    indexes_dir = BASE_DIR / "data" / "indexes"
    emb_path = indexes_dir / "chunk_embeddings.npy"
    units_path = indexes_dir / "chunk_embeddings_ids.json"

    if not emb_path.exists() or not units_path.exists():
        logger.error("Embedding cache files missing! Expected chunk_embeddings.npy and chunk_embeddings_ids.json.")
        return

    logger.info(f"Loading cached embeddings from {emb_path}...")
    embeddings = np.load(emb_path)
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")

    logger.info(f"Loading cached unit metadata from {units_path}...")
    with open(units_path, "r", encoding="utf-8") as f:
        units = json.load(f)

    # 1. Validation checks on cached embeddings
    if len(embeddings) != 13718:
        raise ValueError(f"Expected 13,718 embeddings, found {len(embeddings)}")
    if embeddings.shape[1] != 384:
        raise ValueError(f"Expected 384 dimensions, found {embeddings.shape[1]}")
    if np.isnan(embeddings).any():
        raise ValueError("Embeddings matrix contains NaN values!")
    if np.isinf(embeddings).any():
        raise ValueError("Embeddings matrix contains Inf values!")

    logger.info(f"Vector validation PASSED: {len(embeddings)} vectors, {embeddings.shape[1]} dim, 0 NaNs/Infs.")

    # 2. Build ChromaDB Persistent Collection
    chroma_dir = indexes_dir / "chroma"
    indexer = ChromaDBIndexer(db_dir=chroma_dir, collection_name="research_mind_chunks", dim=384)
    indexer.build_index(embeddings, units, batch_size=500)

    # 3. Verify collection record count
    count = indexer.collection.count()
    logger.info(f"ChromaDB ingestion COMPLETE. Persistent collection '{indexer.collection_name}' count = {count}")
    print("=" * 70)
    print(f"CHROMADB BUILD SUCCESSFUL: {count} records stored at {chroma_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
