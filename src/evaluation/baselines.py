"""
Phase 28 — Baseline Evaluation Framework

Provides a unified evaluation interface for comparing:
1. Baseline 1: LLM Without Retrieval (Direct Generation)
2. Baseline 2: Basic Vector RAG (Dense ChromaDB only)
3. Baseline 3: BM25 Lexical RAG (Lexical BM25 only)
4. Baseline 4: Hybrid RAG (Vector + BM25 without Domain Isolation)
5. Proposed: Full Research Mind System (Domain Scope + Hybrid RRF + Relevance Gate + Claim Grounding + ArXiv Provenance Fallback)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    evaluate_domain_isolation,
    evaluate_answer_completeness,
    evaluate_groundedness,
    evaluate_online_provenance
)


class BaselineEvaluator:
    """Evaluates systems across the 170-question Research Mind benchmark."""

    def __init__(self, benchmark_path: Path):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_dataset = json.load(f)

    def run_benchmark_evaluation(self, system_runner_fn, system_name: str = "Full Research Mind") -> Dict[str, Any]:
        """
        Runs evaluation for a given system function across all 170 benchmark questions.
        """
        results = []
        total_precision_5 = 0.0
        total_recall_5 = 0.0
        total_mrr = 0.0
        total_ndcg_5 = 0.0
        total_domain_isolation = 0.0
        total_leakage_rate = 0.0
        total_completeness = 0.0
        total_groundedness = 0.0
        total_online_provenance = 0.0

        for item in self.benchmark_dataset:
            q_id = item["id"]
            question = item["question"]
            expected_domain = item["expected_domain"]
            expected_intent = item["expected_intent"]
            relevant_pids = item.get("relevant_paper_ids", [])
            category = item.get("category", "single_domain")

            # Execute system runner
            response = system_runner_fn(question)

            # Extract retrieved papers & citations
            retrieved_pids = []
            retrieved_domains = []
            if hasattr(response, "citations") and response.citations:
                for cit in response.citations.values():
                    pid = getattr(cit, "paper_id", "")
                    if pid:
                        retrieved_pids.append(pid)
                    dom = getattr(cit, "domain", "")
                    if dom:
                        retrieved_domains.append(dom)

            allowed_domains = expected_domain if isinstance(expected_domain, list) else [expected_domain]

            # Compute metrics
            p5 = precision_at_k(retrieved_pids, relevant_pids, k=5) if relevant_pids else 1.0
            r5 = recall_at_k(retrieved_pids, relevant_pids, k=5) if relevant_pids else 1.0
            mrr_val = mean_reciprocal_rank(retrieved_pids, relevant_pids) if relevant_pids else 1.0
            ndcg5_val = ndcg_at_k(retrieved_pids, relevant_pids, k=5) if relevant_pids else 1.0

            iso_metrics = evaluate_domain_isolation(retrieved_domains, allowed_domains)
            iso_acc = iso_metrics["domain_isolation_accuracy"]
            leak_rate = iso_metrics["cross_domain_leakage_rate"]

            ans_text = getattr(response, "answer", str(response))
            comp_score = evaluate_answer_completeness(ans_text, expected_intent)

            cits_dict = getattr(response, "citations", {})
            ground_score = evaluate_groundedness(ans_text, cits_dict)
            prov_score = evaluate_online_provenance(cits_dict)

            total_precision_5 += p5
            total_recall_5 += r5
            total_mrr += mrr_val
            total_ndcg_5 += ndcg5_val
            total_domain_isolation += iso_acc
            total_leakage_rate += leak_rate
            total_completeness += comp_score
            total_groundedness += ground_score
            total_online_provenance += prov_score

            results.append({
                "id": q_id,
                "category": category,
                "precision_5": p5,
                "recall_5": r5,
                "mrr": mrr_val,
                "ndcg_5": ndcg5_val,
                "domain_isolation_accuracy": iso_acc,
                "cross_domain_leakage_rate": leak_rate,
                "answer_completeness": comp_score,
                "groundedness": ground_score,
                "online_provenance_accuracy": prov_score
            })

        n = len(self.benchmark_dataset)
        summary = {
            "system_name": system_name,
            "total_evaluated_queries": n,
            "mean_precision_at_5": round(total_precision_5 / n, 4),
            "mean_recall_at_5": round(total_recall_5 / n, 4),
            "mean_mrr": round(total_mrr / n, 4),
            "mean_ndcg_at_5": round(total_ndcg_5 / n, 4),
            "domain_isolation_accuracy": round(total_domain_isolation / n, 4),
            "cross_domain_leakage_rate": round(total_leakage_rate / n, 4),
            "answer_completeness_score": round(total_completeness / n, 4),
            "groundedness_score": round(total_groundedness / n, 4),
            "online_provenance_accuracy": round(total_online_provenance / n, 4)
        }

        return summary
