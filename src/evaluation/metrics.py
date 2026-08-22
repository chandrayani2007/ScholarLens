"""
Phase 28 — Evaluation Metrics Engine

Provides metric calculation utilities for:
1. Retrieval Performance: Precision@K, Recall@K, MRR, NDCG@K
2. Answer Quality: Answer Correctness, Answer Completeness, Groundedness/Faithfulness
3. Citation Accuracy: Citation Precision, Citation Recall, Citation Correctness
4. Research Mind Safeguards:
   - Domain Isolation Accuracy
   - Cross-Domain Leakage Rate
   - Unsupported Claim Rate
   - Online Fallback Accuracy
   - Online Citation Provenance Accuracy
"""

import math
import re
from typing import List, Dict, Set, Any, Optional


def precision_at_k(retrieved: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Compute Precision@K."""
    if not retrieved or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    gt_set = set(ground_truth)
    hits = sum(1 for item in top_k if item in gt_set)
    return hits / float(k)


def recall_at_k(retrieved: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Compute Recall@K."""
    if not ground_truth:
        return 1.0
    top_k = retrieved[:k]
    gt_set = set(ground_truth)
    hits = sum(1 for item in top_k if item in gt_set)
    return hits / float(len(gt_set))


def mean_reciprocal_rank(retrieved: List[str], ground_truth: List[str]) -> float:
    """Compute MRR (Mean Reciprocal Rank)."""
    gt_set = set(ground_truth)
    for rank, item in enumerate(retrieved, 1):
        if item in gt_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    if not ground_truth or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    gt_set = set(ground_truth)

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k, 1):
        rel = 1.0 if item in gt_set else 0.0
        dcg += rel / math.log2(i + 1)

    # IDCG (Ideal DCG)
    idcg = 0.0
    num_relevant = min(len(gt_set), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / math.log2(i + 1)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_domain_isolation(retrieved_domains: List[str], allowed_domains: List[str]) -> Dict[str, float]:
    """
    Evaluates domain isolation & leakage rate.
    allowed_domains: List of domain strings allowed for query scope.
    """
    if not retrieved_domains:
        return {"domain_isolation_accuracy": 1.0, "cross_domain_leakage_rate": 0.0}

    allowed_set = set(allowed_domains)
    leaked_count = sum(1 for d in retrieved_domains if d not in allowed_set)
    total = len(retrieved_domains)

    leakage_rate = leaked_count / total
    isolation_accuracy = 1.0 - leakage_rate

    return {
        "domain_isolation_accuracy": isolation_accuracy,
        "cross_domain_leakage_rate": leakage_rate
    }


def evaluate_answer_completeness(answer: str, intent: str) -> float:
    """
    Evaluates whether answer provides adequate depth for detected intent.
    """
    words = re.findall(r"\b\w+\b", answer)
    word_count = len(words)
    if word_count < 90:
        return 0.5

    intent_clean = intent.strip().lower()
    if intent_clean in ["definition", "mechanism", "explanation", "advantages", "limitations", "comparison"]:
        if word_count >= 110:
            return 1.0
        else:
            return 0.85
    return 1.0


def evaluate_groundedness(answer: str, citations: Dict[str, Any], unsupported_claims_count: int = 0) -> float:
    """
    Evaluates answer groundedness score based on citation alignment & unsupported claims.
    """
    inline_tags = set(re.findall(r"\[([EO]\d+)\]", answer))
    citation_keys = set(citations.keys())

    if not inline_tags and not citations:
        return 0.0

    # Bidirectional alignment score
    missing_cits = len(inline_tags - citation_keys)
    extra_cits = len(citation_keys - inline_tags)

    if missing_cits == 0 and extra_cits == 0 and unsupported_claims_count == 0:
        return 1.0
    else:
        penalty = (missing_cits + extra_cits + unsupported_claims_count) * 0.15
        return max(0.0, 1.0 - penalty)


def evaluate_online_provenance(citations: Dict[str, Any]) -> float:
    """
    Evaluates online ArXiv citation metadata completeness (authors, published, title, url).
    """
    online_cits = [c for c in citations.values() if getattr(c, "source_type", getattr(c, "get", lambda k, default=None: getattr(c, k, default))("source_type")) == "online"]
    if not online_cits:
        return 1.0

    valid_count = 0
    for cit in online_cits:
        paper_id = getattr(cit, "paper_id", "") if hasattr(cit, "paper_id") else cit.get("paper_id", "")
        title = getattr(cit, "title", "") if hasattr(cit, "title") else cit.get("title", "")
        authors = getattr(cit, "authors", []) if hasattr(cit, "authors") else cit.get("authors", [])
        published = getattr(cit, "published", "") if hasattr(cit, "published") else cit.get("published", "")

        if paper_id.startswith("arXiv:") and len(title) > 5 and authors and len(published) >= 4:
            valid_count += 1

    return valid_count / float(len(online_cits))
