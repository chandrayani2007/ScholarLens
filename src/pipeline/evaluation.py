"""
Phase 5 — Retrieval Evaluator Engine (Production Refined)

Calculates:
- Hit Rate@1, Hit Rate@5, Hit Rate@10
- Precision@5, Precision@10
- Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain@10 (nDCG@10) with 0-3 graded relevance

Features:
- Comparative evaluation of ChromaDB, BM25, and Production Filtered Hybrid RRF
- Corrected Recall@K formula with precision-focused ground truth (avg ~19.4 units/query)
- Domain-wise breakdowns (AI, Cybersecurity, Agriculture, Healthcare, Climate, Overall)
- Configurable section filtering via config/retrieval.json
"""

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.pipeline.retrieval import HybridRetriever, RetrievalResult, RetrievalConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MetricSummary:
    hit_at_1: float = 0.0
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def compute_dcg(relevances: List[int], k: int = 10) -> float:
    """Calculate Discounted Cumulative Gain@K using 2^rel - 1 gain formulation."""
    dcg = 0.0
    for j, rel in enumerate(relevances[:k], 1):
        if rel > 0:
            gain = (2.0 ** rel) - 1.0
            discount = math.log2(j + 1.0)
            dcg += gain / discount
    return dcg


def compute_ndcg_at_k(retrieved_relevances: List[int], ground_truth_relevances: List[int], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain@K (nDCG@K)."""
    dcg = compute_dcg(retrieved_relevances, k=k)
    ideal_sorted = sorted(ground_truth_relevances, reverse=True)
    idcg = compute_dcg(ideal_sorted, k=k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


class RetrievalEvaluator:
    """
    Evaluator engine for measuring retrieval systems against ground truth datasets.
    """

    def __init__(
        self,
        queries_path: Optional[Path] = None,
        ground_truth_path: Optional[Path] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        if queries_path is None:
            queries_path = BASE_DIR / "evaluation" / "test_queries.json"
        if ground_truth_path is None:
            ground_truth_path = BASE_DIR / "evaluation" / "ground_truth.json"

        with open(queries_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)

        with open(ground_truth_path, "r", encoding="utf-8") as f:
            self.ground_truth = json.load(f)

        if retriever is None:
            self.retriever = HybridRetriever()
        else:
            self.retriever = retriever

    def evaluate_query(
        self,
        retrieved_results: List[RetrievalResult],
        gt_entries: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate a single query's retrieved results against ground truth mappings.
        Binary relevance threshold: relevance >= 1.
        Graded nDCG uses original 0-3 relevance scores.
        """
        gt_rel_map = {item["unit_id"]: item["relevance"] for item in gt_entries}
        gt_parent_map = {}
        for item in gt_entries:
            pid = item.get("parent_chunk_id")
            if pid and pid not in gt_parent_map:
                gt_parent_map[pid] = item["relevance"]

        retrieved_k = retrieved_results[:top_k]
        retrieved_relevances = []
        for r in retrieved_k:
            rel = gt_rel_map.get(r.unit_id)
            if rel is None:
                rel = gt_parent_map.get(r.parent_chunk_id, 0)
            retrieved_relevances.append(rel)

        while len(retrieved_relevances) < top_k:
            retrieved_relevances.append(0)

        all_gt_relevances = [item["relevance"] for item in gt_entries]
        total_relevant_gt = sum(1 for r in all_gt_relevances if r >= 1)

        # 1. Hit Rates
        hit_1 = 1.0 if any(r >= 1 for r in retrieved_relevances[:1]) else 0.0
        hit_5 = 1.0 if any(r >= 1 for r in retrieved_relevances[:5]) else 0.0
        hit_10 = 1.0 if any(r >= 1 for r in retrieved_relevances[:10]) else 0.0

        # 2. Precision
        rel_in_5 = sum(1 for r in retrieved_relevances[:5] if r >= 1)
        rel_in_10 = sum(1 for r in retrieved_relevances[:10] if r >= 1)
        prec_5 = rel_in_5 / 5.0
        prec_10 = rel_in_10 / 10.0

        # 3. Recall@K (Denominator: total relevant ground-truth units for query)
        rec_5 = (rel_in_5 / total_relevant_gt) if total_relevant_gt > 0 else 0.0
        rec_10 = (rel_in_10 / total_relevant_gt) if total_relevant_gt > 0 else 0.0

        # 4. MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for rank_idx, rel in enumerate(retrieved_relevances, 1):
            if rel >= 1:
                mrr = 1.0 / rank_idx
                break

        # 5. nDCG@10
        ndcg_10 = compute_ndcg_at_k(retrieved_relevances, all_gt_relevances, k=10)

        return {
            "hit_at_1": hit_1,
            "hit_at_5": hit_5,
            "hit_at_10": hit_10,
            "precision_at_5": prec_5,
            "precision_at_10": prec_10,
            "recall_at_5": rec_5,
            "recall_at_10": rec_10,
            "mrr": mrr,
            "ndcg_at_10": ndcg_10,
        }

    def aggregate_metrics(self, query_metrics: List[Dict[str, float]]) -> MetricSummary:
        """Compute mean metrics across a list of query evaluation outputs."""
        if not query_metrics:
            return MetricSummary()

        n = len(query_metrics)
        return MetricSummary(
            hit_at_1=sum(m["hit_at_1"] for m in query_metrics) / n,
            hit_at_5=sum(m["hit_at_5"] for m in query_metrics) / n,
            hit_at_10=sum(m["hit_at_10"] for m in query_metrics) / n,
            precision_at_5=sum(m["precision_at_5"] for m in query_metrics) / n,
            precision_at_10=sum(m["precision_at_10"] for m in query_metrics) / n,
            recall_at_5=sum(m["recall_at_5"] for m in query_metrics) / n,
            recall_at_10=sum(m["recall_at_10"] for m in query_metrics) / n,
            mrr=sum(m["mrr"] for m in query_metrics) / n,
            ndcg_at_10=sum(m["ndcg_at_10"] for m in query_metrics) / n,
        )

    def evaluate_system(
        self,
        mode: str = "rrf",
        enable_section_filtering: bool = True,
        rrf_config: Optional[Tuple[float, float]] = None,
    ) -> Tuple[MetricSummary, Dict[str, MetricSummary], List[Dict[str, Any]]]:
        """
        Evaluate system (mode='chromadb', 'bm25', or 'rrf') across all 50 evaluation queries.
        Returns: (overall_summary, domain_summaries, query_level_details)
        """
        query_results = []
        domain_query_metrics: Dict[str, List[Dict[str, float]]] = {}

        # Backup retriever section filtering setting
        orig_filter_setting = self.retriever.config.enable_section_filtering
        self.retriever.config.enable_section_filtering = enable_section_filtering

        dense_weight, lexical_weight = (1.0, 1.0)
        if rrf_config:
            dense_weight, lexical_weight = rrf_config

        for q in self.queries:
            qid = q["query_id"]
            domain = q["domain"]
            gt_entries = self.ground_truth.get(qid, [])

            if mode == "chromadb":
                query_vec = self.retriever.generator.encode_queries([q["question"]])
                raw = self.retriever.chroma_indexer.search(query_vec[0], k=30)
                if enable_section_filtering:
                    raw = [r for r in raw if not self.retriever._is_excluded_section(r.get("section_name", ""))]
                results = []
                for idx, item in enumerate(raw[:10], 1):
                    results.append(RetrievalResult(
                        rank=idx,
                        unit_id=item["unit_id"],
                        chunk_id=item.get("chunk_id", item["unit_id"]),
                        parent_chunk_id=item.get("parent_chunk_id", item["unit_id"]),
                        paper_id=item["paper_id"],
                        section_id=item["section_id"],
                        section_name=item["section_name"],
                        domain=item["domain"],
                        subtopic=item["subtopic"],
                        page_start=item["page_start"],
                        page_end=item["page_end"],
                        text=item["text"],
                        token_count=item.get("token_count", 0),
                        dense_score=item.get("score"),
                        rrf_score=round(1.0 / (60 + idx), 6),
                        retrieval_methods=["ChromaDB"],
                    ))

            elif mode == "bm25":
                raw = self.retriever.bm25_indexer.search(q["question"], k=30)
                if enable_section_filtering:
                    raw = [r for r in raw if not self.retriever._is_excluded_section(r.get("section_name", ""))]
                results = []
                for idx, item in enumerate(raw[:10], 1):
                    results.append(RetrievalResult(
                        rank=idx,
                        unit_id=item["unit_id"],
                        chunk_id=item.get("chunk_id", item["unit_id"]),
                        parent_chunk_id=item.get("parent_chunk_id", item["unit_id"]),
                        paper_id=item["paper_id"],
                        section_id=item["section_id"],
                        section_name=item["section_name"],
                        domain=item["domain"],
                        subtopic=item["subtopic"],
                        page_start=item["page_start"],
                        page_end=item["page_end"],
                        text=item["text"],
                        token_count=item.get("token_count", 0),
                        bm25_score=item.get("score"),
                        rrf_score=round(1.0 / (60 + idx), 6),
                        retrieval_methods=["BM25"],
                    ))

            else:  # Hybrid RRF
                orig_dw = self.retriever.config.dense_weight
                orig_lw = self.retriever.config.lexical_weight
                self.retriever.config.dense_weight = dense_weight
                self.retriever.config.lexical_weight = lexical_weight

                raw_results = self.retriever.retrieve(q["question"], top_k=10)
                results = raw_results[:10]

                self.retriever.config.dense_weight = orig_dw
                self.retriever.config.lexical_weight = orig_lw

            metrics = self.evaluate_query(results, gt_entries, top_k=10)

            query_record = {
                "query_id": qid,
                "domain": domain,
                "subtopic": q["subtopic"],
                "question": q["question"],
                "difficulty": q["difficulty"],
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
                "retrieved_top_unit": results[0].unit_id if results else None,
                "retrieved_top_section": results[0].section_name if results else None,
            }
            query_results.append(query_record)
            domain_query_metrics.setdefault(domain, []).append(metrics)

        # Restore original section filtering setting
        self.retriever.config.enable_section_filtering = orig_filter_setting

        all_metrics = [q["metrics"] for q in query_results]
        overall_summary = self.aggregate_metrics(all_metrics)

        domain_summaries = {}
        for d, d_metrics in domain_query_metrics.items():
            domain_summaries[d] = self.aggregate_metrics(d_metrics)

        return overall_summary, domain_summaries, query_results
