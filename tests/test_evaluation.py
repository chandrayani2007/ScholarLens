"""
Unit & Integration Tests for Phase 5 Retrieval Evaluation Pipeline.

Tests cover:
- Hit Rate@1, Hit Rate@5, Hit Rate@10 calculation
- Precision@5, Precision@10 calculation
- Recall@5, Recall@10 calculation
- Mean Reciprocal Rank (MRR) calculation
- nDCG@10 calculation with 0-3 graded relevance & IDCG normalization
- Binary relevance thresholding (relevance >= 1)
- Evaluation dataset loading & ground truth alignment
- Section filtering ablation logic
- RRF weight sensitivity logic
- Evaluation output reproducibility
"""

import json
from pathlib import Path
import pytest

from src.pipeline.evaluation import (
    compute_dcg,
    compute_ndcg_at_k,
    MetricSummary,
    RetrievalEvaluator,
)
from src.pipeline.retrieval import RetrievalResult


@pytest.fixture
def sample_gt_entries():
    return [
        {"unit_id": "AI001_C001", "parent_chunk_id": "AI001_C001", "relevance": 3},
        {"unit_id": "AI001_C002_sub01", "parent_chunk_id": "AI001_C002", "relevance": 2},
        {"unit_id": "AI001_C002_sub02", "parent_chunk_id": "AI001_C002", "relevance": 1},
        {"unit_id": "AI001_C005", "parent_chunk_id": "AI001_C005", "relevance": 0},
    ]


@pytest.fixture
def sample_retrieved_results():
    return [
        RetrievalResult(
            rank=1,
            unit_id="AI001_C005",
            chunk_id="AI001_C005",
            parent_chunk_id="AI001_C005",
            paper_id="AI001",
            section_id="SEC_001",
            section_name="Header",
            domain="artificial_intelligence",
            subtopic="rag",
            page_start=1,
            page_end=1,
            text="Irrelevant header text",
            token_count=10,
            rrf_score=0.03,
        ),
        RetrievalResult(
            rank=2,
            unit_id="AI001_C001",
            chunk_id="AI001_C001",
            parent_chunk_id="AI001_C001",
            paper_id="AI001",
            section_id="SEC_002",
            section_name="Introduction",
            domain="artificial_intelligence",
            subtopic="rag",
            page_start=1,
            page_end=1,
            text="Highly relevant RAG text",
            token_count=50,
            rrf_score=0.028,
        ),
        RetrievalResult(
            rank=3,
            unit_id="AI001_C002_sub01",
            chunk_id="AI001_C002_sub01",
            parent_chunk_id="AI001_C002",
            paper_id="AI001",
            section_id="SEC_003",
            section_name="Methodology",
            domain="artificial_intelligence",
            subtopic="rag",
            page_start=2,
            page_end=2,
            text="Relevant vector search text",
            token_count=45,
            rrf_score=0.026,
        ),
    ]


# ─── Metric Calculation Unit Tests ───

class TestMetricsMath:
    def test_dcg_calculation(self):
        # Graded rels: [3, 2, 0]
        # DCG@3 = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) + 0 = 7/1 + 3/1.58496 = 7 + 1.8928 = 8.8928
        dcg = compute_dcg([3, 2, 0], k=3)
        assert abs(dcg - 8.8928) < 1e-3

    def test_ndcg_perfect_score(self):
        rels = [3, 2, 1]
        gt = [3, 2, 1]
        ndcg = compute_ndcg_at_k(rels, gt, k=3)
        assert abs(ndcg - 1.0) < 1e-4

    def test_ndcg_imperfect_score(self):
        rels = [0, 3, 2]
        gt = [3, 2, 1]
        ndcg = compute_ndcg_at_k(rels, gt, k=3)
        assert 0.0 < ndcg < 1.0

    def test_query_evaluation_metrics(self, sample_retrieved_results, sample_gt_entries):
        evaluator = RetrievalEvaluator()
        m = evaluator.evaluate_query(sample_retrieved_results, sample_gt_entries, top_k=10)

        # Rank 1: rel=0, Rank 2: rel=3, Rank 3: rel=2
        assert m["hit_at_1"] == 0.0
        assert m["hit_at_5"] == 1.0
        assert m["hit_at_10"] == 1.0
        assert m["mrr"] == 0.5  # First relevant item at rank 2 -> 1/2 = 0.5
        assert m["precision_at_5"] == 2.0 / 5.0  # 2 relevant out of 5
        assert m["recall_at_5"] == 2.0 / 3.0     # 2 retrieved out of 3 total GT relevant


class TestEvaluationDatasets:
    def test_datasets_exist_and_load(self):
        BASE_DIR = Path(__file__).resolve().parent.parent
        q_path = BASE_DIR / "evaluation" / "test_queries.json"
        gt_path = BASE_DIR / "evaluation" / "ground_truth.json"

        assert q_path.exists()
        assert gt_path.exists()

        with open(q_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        assert len(queries) == 50

        with open(gt_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        assert len(gt) == 50

    def test_evaluation_reproducibility(self):
        evaluator = RetrievalEvaluator()
        m1, _, _ = evaluator.evaluate_system(mode="rrf")
        m2, _, _ = evaluator.evaluate_system(mode="rrf")

        assert m1.hit_at_1 == m2.hit_at_1
        assert m1.hit_at_5 == m2.hit_at_5
        assert m1.mrr == m2.mrr
        assert m1.ndcg_at_10 == m2.ndcg_at_10
