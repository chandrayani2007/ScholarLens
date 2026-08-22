"""
Phase 4 / 26 — Hybrid Retrieval Engine with Intent-Aware Query Expansion & Multi-Domain Isolation

Features:
1. IntentQueryReformulator expanding research queries based on detected intent (Limitation, Advantage, Mechanism, etc.).
2. Dual-stream ChromaDB dense retrieval & BM25 lexical retrieval using both raw query and intent-expanded query.
3. Reciprocal Rank Fusion (RRF) using rank positions (1 / (k + r)) across candidate streams.
4. Provenance retention (unit_id, parent_chunk_id, paper_id, section_id, page_start, page_end).
5. Parent chunk deduplication without provenance destruction.
6. Multi-Domain Scope Support: Accepts allowed_domains list (e.g. ["artificial_intelligence", "healthcare"]).
   Candidates outside allowed_domains are strictly excluded at EVERY retrieval stage.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import numpy as np

from src.pipeline.embedding import IndexingConfig, EmbeddingGenerator
from src.pipeline.indexing import BM25Indexer
from src.pipeline.chroma_indexer import ChromaDBIndexer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_FILTERS = {"domain", "subtopic", "paper_id", "section_name"}

DOMAIN_PREFIX_MAP = {
    "artificial_intelligence": "AI",
    "cybersecurity": "CY",
    "agriculture": "AG",
    "healthcare": "HC",
    "climate": "CL",
}

INTENT_EXPANSION_KEYWORDS = {
    "Limitation": ["limitations", "challenges", "drawbacks", "weaknesses", "failures", "bottlenecks", "constraints", "evaluation", "risks", "errors", "noise"],
    "Challenge": ["challenges", "difficulties", "barriers", "limitations", "drawbacks", "bottlenecks", "complexities"],
    "Advantage": ["advantages", "benefits", "improvements", "effectiveness", "performance", "superiority", "gains", "optimization"],
    "Mechanism": ["method", "architecture", "workflow", "mechanism", "implementation", "process", "pipeline", "algorithm", "step"],
    "Process": ["process", "workflow", "stages", "pipeline", "execution", "steps", "procedure"],
    "Evaluation": ["evaluation", "benchmarks", "metrics", "measurement", "assessment", "validation", "empirical", "datasets"],
    "Application": ["applications", "deployment", "use case", "real-world", "clinical", "practical", "field", "implementation"],
    "Comparison": ["comparison", "versus", "differences", "trade-offs", "comparative analysis", "relative"],
    "Definition": ["definition", "concept", "overview", "fundamentals", "principles", "introduction"],
    "Role": ["role", "impact", "importance", "influence", "contribution", "significance"],
    "Cause": ["causes", "reasons", "underlying", "factors", "triggers", "drivers"],
    "Effect": ["effects", "impact", "outcomes", "consequences", "results", "influence"],
    "Detection": ["detection", "identification", "discovery", "monitoring", "screening", "diagnosis"],
    "Prediction": ["prediction", "forecasting", "estimation", "projections", "modeling"],
}


class IntentQueryReformulator:
    """Expands research queries using detected intent terms to boost specific subtopic passages."""

    @staticmethod
    def expand_query(query: str, intent: Optional[str] = None) -> str:
        if not query or not intent or intent not in INTENT_EXPANSION_KEYWORDS:
            return query

        keywords = INTENT_EXPANSION_KEYWORDS[intent]
        expansion_str = " ".join(keywords[:5])
        return f"{query} {expansion_str}"


def is_bibliography_chunk(text: str) -> bool:
    """Detect if chunk text is predominantly a reference list or bibliography metadata."""
    if not text or len(text.strip()) < 30:
        return True
    
    ref_brackets = re.findall(r"\[\d{1,3}\]", text)
    if len(ref_brackets) >= 3:
        return True

    bib_keywords = ["doi:", "vol.", "pp.", "et al.", "isbn", "proceedings of", "journal of", "springer", "ieee", "arxiv:"]
    text_lower = text.lower()
    match_count = sum(1 for kw in bib_keywords if kw in text_lower)
    if match_count >= 2:
        return True

    return False


@dataclass
class RetrievalConfig:
    dense_backend: str = "chromadb"
    chroma_db_dir: str = "data/indexes/chroma"
    chroma_collection: str = "research_mind_chunks"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    top_k_dense: int = 20
    top_k_bm25: int = 20
    rrf_k: int = 60
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    final_top_k: int = 10
    deduplicate_parent_chunks: bool = True
    enable_section_filtering: bool = True
    excluded_sections: List[str] = field(default_factory=lambda: ["Header", "References", "Bibliography", "Acknowledgements", "Literature Cited"])

    @classmethod
    def from_file(cls, config_path: Path) -> "RetrievalConfig":
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    rank: int
    unit_id: str
    chunk_id: str
    parent_chunk_id: str
    paper_id: str
    section_id: str
    section_name: str
    domain: str
    subtopic: str
    page_start: int
    page_end: int
    text: str
    token_count: int
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0
    retrieval_methods: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridRetriever:
    """
    Standalone hybrid retrieval service combining ChromaDB dense vector search and BM25 lexical search via RRF.
    Supports intent-aware query expansion and strict multi-domain scope isolation.
    """

    def __init__(
        self,
        config: Optional[RetrievalConfig] = None,
        chroma_indexer: Optional[ChromaDBIndexer] = None,
        bm25_indexer: Optional[BM25Indexer] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        index_dir: Optional[Path] = None,
    ):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        if config is None:
            config_path = BASE_DIR / "config" / "retrieval.json"
            if config_path.exists():
                self.config = RetrievalConfig.from_file(config_path)
            else:
                self.config = RetrievalConfig()
        else:
            self.config = config

        if index_dir is None:
            index_dir = BASE_DIR / "data" / "indexes"

        if embedding_generator is None:
            idx_config = IndexingConfig(
                embedding_model=self.config.embedding_model,
                embedding_dim=self.config.embedding_dim,
                query_instruction=self.config.query_instruction,
            )
            self.generator = EmbeddingGenerator(idx_config)
        else:
            self.generator = embedding_generator

        if chroma_indexer is None:
            chroma_dir = index_dir / "chroma"
            self.chroma_indexer = ChromaDBIndexer(
                db_dir=chroma_dir,
                collection_name="research_mind_chunks",
                dim=self.config.embedding_dim
            )
            if (chroma_dir / "chroma.sqlite3").exists():
                self.chroma_indexer.load()
        else:
            self.chroma_indexer = chroma_indexer

        if bm25_indexer is None:
            self.bm25_indexer = BM25Indexer()
            bm25_path = index_dir / "bm25.pkl"
            if bm25_path.exists():
                self.bm25_indexer.load(bm25_path)
        else:
            self.bm25_indexer = bm25_indexer

    def _is_excluded_section(self, section_name: str, text: str = "") -> bool:
        """Check if section_name matches any excluded section in config or text is a bibliography chunk."""
        if not self.config.enable_section_filtering:
            return False
        
        if self.config.excluded_sections:
            sec_lower = section_name.lower()
            if any(ex.lower() in sec_lower for ex in self.config.excluded_sections):
                return True

        if text and is_bibliography_chunk(text):
            return True

        return False

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        allowed_domains: Optional[List[str]] = None,
        intent: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Execute hybrid retrieval with intent-aware query expansion, ChromaDB, BM25, RRF fusion,
        and strict multi-domain scope pre-filtering.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace-only")

        if filters:
            for k in filters:
                if k not in ALLOWED_FILTERS:
                    raise ValueError(f"Invalid filter key '{k}'. Allowed filter keys: {sorted(ALLOWED_FILTERS)}")

        effective_top_k = top_k if top_k is not None else self.config.final_top_k
        candidate_pool_multiplier = 3 if self.config.enable_section_filtering else 2

        domain_pool: Set[str] = set()
        if allowed_domains:
            domain_pool.update(allowed_domains)
        elif filters and filters.get("domain"):
            d_val = filters["domain"]
            if isinstance(d_val, (list, tuple, set)):
                domain_pool.update(d_val)
            elif str(d_val).lower() not in ("all", "all domains", "none", ""):
                domain_pool.add(str(d_val))

        expected_prefixes = {DOMAIN_PREFIX_MAP[d] for d in domain_pool if d in DOMAIN_PREFIX_MAP}

        def is_domain_allowed(item: Dict[str, Any]) -> bool:
            if not domain_pool:
                return True
            item_dom = item.get("domain", "")
            item_paper = item.get("paper_id", "")
            if item_dom not in domain_pool:
                return False
            if expected_prefixes and not any(item_paper.startswith(p) for p in expected_prefixes):
                return False
            return True

        search_filters = dict(filters) if filters else {}
        if domain_pool:
            if len(domain_pool) == 1:
                search_filters["domain"] = list(domain_pool)[0]
            else:
                search_filters["domain"] = list(domain_pool)

        # Generate raw query and intent-expanded query
        expanded_query = IntentQueryReformulator.expand_query(query, intent)

        # 1. ChromaDB Dense Retrieval (Dual Stream: Raw + Expanded)
        raw_chroma_candidates_1 = []
        raw_chroma_candidates_2 = []
        if self.config.top_k_dense > 0:
            query_vecs = self.generator.encode_queries([query, expanded_query])
            raw_chroma_candidates_1 = self.chroma_indexer.search(
                query_vector=query_vecs[0],
                k=self.config.top_k_dense * candidate_pool_multiplier,
                filters=search_filters
            )
            raw_chroma_candidates_2 = self.chroma_indexer.search(
                query_vector=query_vecs[1],
                k=self.config.top_k_dense * candidate_pool_multiplier,
                filters=search_filters
            )

        # 2. BM25 Lexical Retrieval (Dual Stream: Raw + Expanded)
        raw_bm25_results_1 = []
        raw_bm25_results_2 = []
        if self.config.top_k_bm25 > 0:
            raw_bm25_results_1 = self.bm25_indexer.search(
                query_str=query,
                k=self.config.top_k_bm25 * candidate_pool_multiplier,
                filters=search_filters
            )
            raw_bm25_results_2 = self.bm25_indexer.search(
                query_str=expanded_query,
                k=self.config.top_k_bm25 * candidate_pool_multiplier,
                filters=search_filters
            )

        chroma_candidates_1 = [
            r for r in raw_chroma_candidates_1
            if is_domain_allowed(r) and not self._is_excluded_section(r.get("section_name", ""), r.get("text", ""))
        ][: self.config.top_k_dense]

        chroma_candidates_2 = [
            r for r in raw_chroma_candidates_2
            if is_domain_allowed(r) and not self._is_excluded_section(r.get("section_name", ""), r.get("text", ""))
        ][: self.config.top_k_dense]

        bm25_candidates_1 = [
            r for r in raw_bm25_results_1
            if is_domain_allowed(r) and not self._is_excluded_section(r.get("section_name", ""), r.get("text", ""))
        ][: self.config.top_k_bm25]

        bm25_candidates_2 = [
            r for r in raw_bm25_results_2
            if is_domain_allowed(r) and not self._is_excluded_section(r.get("section_name", ""), r.get("text", ""))
        ][: self.config.top_k_bm25]

        # 3. Multi-Stream Reciprocal Rank Fusion (RRF)
        fused_map: Dict[str, Dict[str, Any]] = {}

        def add_stream(candidates: List[Dict[str, Any]], method_name: str):
            for rank_pos, item in enumerate(candidates, 1):
                unit_id = item["unit_id"]
                contrib = (1.0 if "Chroma" in method_name else 1.0) / (self.config.rrf_k + rank_pos)

                if unit_id not in fused_map:
                    fused_map[unit_id] = {
                        "item": item,
                        "dense_score": item.get("score") if "Chroma" in method_name else None,
                        "bm25_score": item.get("score") if "BM25" in method_name else None,
                        "rrf_score": contrib,
                        "methods": [method_name],
                    }
                else:
                    fused_map[unit_id]["rrf_score"] += contrib
                    if "Chroma" in method_name and fused_map[unit_id]["dense_score"] is None:
                        fused_map[unit_id]["dense_score"] = item.get("score")
                    if "BM25" in method_name and fused_map[unit_id]["bm25_score"] is None:
                        fused_map[unit_id]["bm25_score"] = item.get("score")
                    base_m = "ChromaDB" if "Chroma" in method_name else "BM25"
                    if base_m not in fused_map[unit_id]["methods"]:
                        fused_map[unit_id]["methods"].append(base_m)

        add_stream(chroma_candidates_1, "ChromaDB")
        add_stream(chroma_candidates_2, "ChromaDB_Expanded")
        add_stream(bm25_candidates_1, "BM25")
        add_stream(bm25_candidates_2, "BM25_Expanded")

        candidates_list = []
        for unit_id, entry in fused_map.items():
            if is_domain_allowed(entry["item"]):
                candidates_list.append(entry)

        # 4. Deduplication / Parent Chunk Grouping
        if self.config.deduplicate_parent_chunks:
            grouped: Dict[str, Dict[str, Any]] = {}
            for entry in candidates_list:
                parent_id = entry["item"]["parent_chunk_id"]
                if parent_id not in grouped:
                    grouped[parent_id] = entry
                else:
                    existing = grouped[parent_id]
                    if entry["rrf_score"] > existing["rrf_score"]:
                        combined_methods = list(dict.fromkeys(existing["methods"] + entry["methods"]))
                        entry["methods"] = combined_methods
                        grouped[parent_id] = entry
                    else:
                        combined_methods = list(dict.fromkeys(existing["methods"] + entry["methods"]))
                        existing["methods"] = combined_methods

            final_candidates = list(grouped.values())
        else:
            final_candidates = candidates_list

        final_candidates.sort(key=lambda x: (-x["rrf_score"], x["item"]["unit_id"]))

        if domain_pool:
            final_candidates = [c for c in final_candidates if is_domain_allowed(c["item"])]

        top_candidates = final_candidates[:effective_top_k]

        results = []
        for rank_idx, entry in enumerate(top_candidates, 1):
            item = entry["item"]
            methods = [m.replace("_Expanded", "") for m in entry["methods"]]
            methods = list(dict.fromkeys(methods))
            if ("ChromaDB" in methods or "FAISS" in methods) and "BM25" in methods:
                methods_tag = ["both"]
            else:
                methods_tag = methods

            unit_id = item.get("unit_id", "")
            chunk_id = item.get("chunk_id", unit_id)
            parent_chunk_id = item.get("parent_chunk_id", chunk_id)

            r = RetrievalResult(
                rank=rank_idx,
                unit_id=unit_id,
                chunk_id=chunk_id,
                parent_chunk_id=parent_chunk_id,
                paper_id=item["paper_id"],
                section_id=item["section_id"],
                section_name=item["section_name"],
                domain=item["domain"],
                subtopic=item["subtopic"],
                page_start=item["page_start"],
                page_end=item["page_end"],
                text=item["text"],
                token_count=item.get("token_count", 0),
                dense_score=entry["dense_score"],
                bm25_score=entry["bm25_score"],
                rrf_score=entry["rrf_score"],
                retrieval_methods=methods_tag,
            )
            results.append(r)

        return results
