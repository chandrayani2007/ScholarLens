"""
Phase 28.5 — 1,000-Paper Retrieval Coverage & Domain Evaluation Test Suite

Verifies:
1. Deterministically evaluates retrieval across all 1,000 papers (AI001-AI200, CY001-CY200, AG001-AG200, HC001-HC200, CL001-CL200).
2. Calculates Top-1 Accuracy, Top-5 Recall, Top-10 Recall, MRR, and Retrieval Coverage.
3. Exports evaluation/results/1000_paper_coverage.csv and evaluation/results/domain_retrieval_coverage.csv.
"""

import csv
import json
import re
import pytest
from pathlib import Path
from collections import defaultdict

from src.pipeline.rag import RAGPipeline
from src.pipeline.retrieval import HybridRetriever, RetrievalConfig
from src.pipeline.embedding import EmbeddingGenerator, IndexingConfig
from src.pipeline.indexing import BM25Indexer
from src.pipeline.chroma_indexer import ChromaDBIndexer

BASE_DIR = Path(r"c:\Users\nugur\Desktop\researchmind")
PAPERS_JSON_PATH = BASE_DIR / "data" / "metadata" / "papers.json"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class Test1000PaperRetrievalCoverage:

    @pytest.fixture(scope="class")
    def retriever_and_papers(self):
        with open(PAPERS_JSON_PATH, "r", encoding="utf-8") as f:
            papers = json.load(f)

        config_path = BASE_DIR / "config" / "indexing.json"
        indexing_config = IndexingConfig.from_file(config_path)
        generator = EmbeddingGenerator(indexing_config)
        chroma_indexer = ChromaDBIndexer(db_dir=BASE_DIR / "data" / "indexes" / "chroma")
        bm25_indexer = BM25Indexer()
        bm25_indexer.load(BASE_DIR / "data" / "indexes" / "bm25.pkl")

        retrieval_config = RetrievalConfig(top_k_dense=20, top_k_bm25=20, rrf_k=60)
        retriever = HybridRetriever(
            config=retrieval_config,
            embedding_generator=generator,
            chroma_indexer=chroma_indexer,
            bm25_indexer=bm25_indexer,
        )
        return retriever, papers

    def test_full_1000_paper_retrieval_coverage(self, retriever_and_papers):
        retriever, papers = retriever_and_papers
        assert len(papers) == 1000, f"Expected 1,000 papers, found {len(papers)}"

        results_by_paper = []
        domain_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top5": 0, "top10": 0, "mrr_sum": 0.0})

        for idx, paper in enumerate(papers, 1):
            paper_id = paper["paper_id"]
            domain = paper["domain"]
            title = paper["title"]
            abstract = paper.get("abstract", "")

            # Build query from title + main abstract keywords
            clean_title = re.sub(r"[^\w\s]", " ", title).strip()
            query_str = clean_title

            # Retrieve top 10 candidates with domain scoping
            candidates = retriever.retrieve(query=query_str, allowed_domains=[domain], top_k=10)
            retrieved_pids = [c.paper_id for c in candidates]

            rank = None
            if paper_id in retrieved_pids:
                rank = retrieved_pids.index(paper_id) + 1

            retrieved_bool = rank is not None
            top1 = rank == 1
            top5 = rank is not None and rank <= 5
            top10 = rank is not None and rank <= 10
            mrr_val = (1.0 / rank) if rank else 0.0

            # Record per-paper CSV row
            results_by_paper.append({
                "paper_id": paper_id,
                "domain": domain,
                "title": title[:80],
                "retrieved": retrieved_bool,
                "rank": rank if rank else "None",
                "mrr": round(mrr_val, 4)
            })

            # Aggregate domain stats
            d = domain_stats[domain]
            d["total"] += 1
            if top1: d["top1"] += 1
            if top5: d["top5"] += 1
            if top10: d["top10"] += 1
            d["mrr_sum"] += mrr_val

        # 1. Save 1000_paper_coverage.csv
        csv_1000_path = RESULTS_DIR / "1000_paper_coverage.csv"
        with open(csv_1000_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["paper_id", "domain", "title", "retrieved", "rank", "mrr"])
            writer.writeheader()
            writer.writerows(results_by_paper)

        # 2. Compute per-domain metrics & Save domain_retrieval_coverage.csv
        domain_rows = []
        overall_total = 1000
        overall_top1 = sum(d["top1"] for d in domain_stats.values())
        overall_top5 = sum(d["top5"] for d in domain_stats.values())
        overall_top10 = sum(d["top10"] for d in domain_stats.values())
        overall_mrr = sum(d["mrr_sum"] for d in domain_stats.values()) / float(overall_total)

        for dom, d in sorted(domain_stats.items()):
            n = d["total"]
            t1_acc = d["top1"] / float(n)
            t5_rec = d["top5"] / float(n)
            t10_rec = d["top10"] / float(n)
            mrr_avg = d["mrr_sum"] / float(n)
            cov_rate = t10_rec

            domain_rows.append({
                "domain": dom,
                "number_of_papers": n,
                "queries_tested": n,
                "top1_accuracy": round(t1_acc, 4),
                "top5_recall": round(t5_rec, 4),
                "top10_recall": round(t10_rec, 4),
                "mrr": round(mrr_avg, 4),
                "retrieval_coverage": round(cov_rate, 4)
            })

        csv_domain_path = RESULTS_DIR / "domain_retrieval_coverage.csv"
        with open(csv_domain_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["domain", "number_of_papers", "queries_tested", "top1_accuracy", "top5_recall", "top10_recall", "mrr", "retrieval_coverage"])
            writer.writeheader()
            writer.writerows(domain_rows)

        top10_recall_overall = overall_top10 / float(overall_total)
        print(f"\n========================================================")
        print(f"=== 1,000-PAPER RETRIEVAL COVERAGE TEST RESULTS ===")
        print(f"========================================================")
        print(f"Total Papers Tested:    {overall_total}")
        print(f"Top-1 Accuracy:        {overall_top1 / float(overall_total):.4f}")
        print(f"Top-5 Recall:          {overall_top5 / float(overall_total):.4f}")
        print(f"Top-10 Recall:         {top10_recall_overall:.4f}")
        print(f"Overall MRR:           {overall_mrr:.4f}")
        print(f"Paper Retrieval Cov:   {top10_recall_overall * 100:.2f}%")
        print(f"========================================================")

        assert top10_recall_overall > 0.0, "Top-10 recall must be greater than 0!"
