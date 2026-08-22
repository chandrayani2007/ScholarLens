"""
Phase 4 — ChromaDB Dense Vector Indexer

Manages:
1. Persistent local ChromaDB database at data/indexes/chroma/.
2. Collection 'research_mind_chunks' with Cosine distance metric.
3. Batch ingestion of 13,718 vector embeddings, texts, and full provenance metadata.
4. Cosine similarity query execution & metadata filtering.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ChromaDBIndexer:
    """
    ChromaDB dense vector index manager with persistent storage and metadata filtering.
    """

    def __init__(
        self,
        db_dir: Optional[Path] = None,
        collection_name: str = "research_mind_chunks",
        dim: int = 384,
    ):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        if db_dir is None:
            db_dir = BASE_DIR / "data" / "indexes" / "chroma"

        self.db_dir = db_dir
        self.collection_name = collection_name
        self.dim = dim
        self.db_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB PersistentClient at {self.db_dir}")
        self.client = chromadb.PersistentClient(path=str(self.db_dir))
        self.collection = None

    def get_or_create_collection(self):
        """Get or create the target ChromaDB collection with Cosine distance metric."""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        return self.collection

    def build_index(
        self,
        embeddings: np.ndarray,
        units: List[Dict[str, Any]],
        batch_size: int = 500
    ) -> None:
        """
        Populate ChromaDB collection with 13,718 vector embeddings, document texts,
        and full provenance metadata.
        """
        if len(embeddings) != len(units):
            raise ValueError(f"Embeddings count ({len(embeddings)}) != units count ({len(units)})")
        if len(embeddings) > 0 and embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) != configured dimension ({self.dim})")

        # Get or recreate collection
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted existing collection '{self.collection_name}' for clean rebuild")
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        total_units = len(units)
        logger.info(f"Ingesting {total_units} records into ChromaDB collection '{self.collection_name}'...")

        for i in range(0, total_units, batch_size):
            batch_units = units[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size].tolist()

            ids = []
            documents = []
            metadatas = []

            for u in batch_units:
                chunk_id = u.get("chunk_id", "")
                unit_id = u.get("unit_id", chunk_id)
                parent_chunk_id = u.get("parent_chunk_id", chunk_id)

                ids.append(unit_id)
                documents.append(u["text"])
                metadatas.append({
                    "unit_id": unit_id,
                    "parent_chunk_id": parent_chunk_id,
                    "chunk_id": chunk_id,
                    "paper_id": str(u["paper_id"]),
                    "section_id": str(u["section_id"]),
                    "section_name": str(u["section_name"]),
                    "domain": str(u["domain"]),
                    "subtopic": str(u["subtopic"]),
                    "page_start": int(u["page_start"]),
                    "page_end": int(u["page_end"]),
                    "is_sub_chunk": bool(u.get("is_sub_chunk", False)),
                    "token_count": int(u.get("token_count", 0)),
                })

            self.collection.add(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas
            )

        logger.info(f"Successfully ingested {self.collection.count()} records into ChromaDB")

    def load(self):
        """Load existing persistent collection."""
        self.collection = self.client.get_collection(name=self.collection_name)
        logger.info(f"Loaded persistent ChromaDB collection '{self.collection_name}' with {self.collection.count()} records")

    def _format_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Convert metadata filter dictionary to ChromaDB 'where' clause format."""
        if not filters:
            return None

        clean_filters = {k: v for k, v in filters.items() if v is not None}
        if not clean_filters:
            return None

        conditions = []
        for k, v in clean_filters.items():
            if isinstance(v, (list, set, tuple)):
                conditions.append({k: {"$in": list(v)}})
            else:
                conditions.append({k: {"$eq": v}})

        if len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search ChromaDB collection given query vector and optional metadata filters.
        Returns list of result dicts containing converted cosine similarity score and complete metadata.
        """
        if self.collection is None:
            self.load()

        if query_vector.ndim == 1:
            query_vec_list = [query_vector.tolist()]
        else:
            query_vec_list = query_vector.tolist()

        where_clause = self._format_where_clause(filters)

        query_args = {
            "query_embeddings": query_vec_list,
            "n_results": min(k, self.collection.count()),
            "include": ["metadatas", "documents", "distances"]
        }
        if where_clause:
            query_args["where"] = where_clause

        res = self.collection.query(**query_args)

        results = []
        if not res or not res.get("ids") or len(res["ids"][0]) == 0:
            return results

        ids = res["ids"][0]
        documents = res["documents"][0]
        metadatas = res["metadatas"][0]
        distances = res["distances"][0]

        for rank_idx, (rec_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
            # Convert ChromaDB Cosine distance to similarity score: similarity = 1 - distance
            similarity_score = float(1.0 - dist)

            item = dict(meta)
            item["text"] = doc
            item["score"] = similarity_score
            item["distance"] = float(dist)
            item["chroma_index"] = rank_idx
            results.append(item)

        return results
