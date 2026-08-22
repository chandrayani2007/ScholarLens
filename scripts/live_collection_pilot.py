import json
import logging
from pathlib import Path
from src.connectors.arxiv import ArxivConnector
from src.pipeline.ingestion import PaperIngestionPipeline
from src.storage.metadata_store import MetadataStore

logging.basicConfig(level=logging.INFO)

def run_pilot():
    connector = ArxivConnector()
    metadata_store = MetadataStore()
    pipeline = PaperIngestionPipeline(metadata_store=metadata_store)

    print("=================================================================")
    print("RUNNING LIVE PILOT 1: Discovering and Ingesting 3 AI Papers")
    print("=================================================================")

    candidates = connector.search_candidates(
        query="retrieval-augmented generation RAG",
        domain="artificial_intelligence",
        subtopic="retrieval_augmented_generation",
        max_results=5
    )

    accepted_count = 0
    accepted_papers = []

    for cand in candidates:
        if accepted_count >= 3:
            break

        print(f"\nProcessing Candidate: '{cand.title}' (arXiv:{cand.arxiv_id})")
        accepted, metadata, msg = pipeline.process_candidate(cand)

        if accepted and metadata:
            accepted_count += 1
            accepted_papers.append(metadata)
            print(f" -> ACCEPTED: Assigned {metadata.paper_id}")
            print(f"    Path: {metadata.local_path}")
            print(f"    Hash: {metadata.file_hash[:16]}...")
        else:
            print(f" -> SKIPPED/REJECTED: {msg}")

    print("\n=================================================================")
    print(f"PILOT 1 COMPLETE: Accepted {accepted_count} papers.")
    print("=================================================================\n")

    # RERUN TEST
    print("=================================================================")
    print("RUNNING PILOT 2 (RERUN TEST): Re-ingesting Same Candidates")
    print("=================================================================")

    metadata_store.reload()  # Ensure store is reloaded from papers.json
    rerun_pipeline = PaperIngestionPipeline(metadata_store=metadata_store)

    rerun_accepted_count = 0
    for cand in candidates[:accepted_count]:
        print(f"\nRe-processing Candidate: '{cand.title}'")
        accepted, metadata, msg = rerun_pipeline.process_candidate(cand)
        if accepted:
            rerun_accepted_count += 1
            print(f" -> ERROR: Unexpectedly accepted as {metadata.paper_id}")
        else:
            print(f" -> RESUMED (Skipped Duplicate): {msg}")

    print("\n=================================================================")
    print(f"PILOT RERUN COMPLETE: New papers accepted during rerun = {rerun_accepted_count}")
    print("=================================================================")

if __name__ == "__main__":
    run_pilot()
