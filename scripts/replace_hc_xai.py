"""
Targeted replacement of HC046, HC047, HC048 with healthcare-specific XAI papers.
Uses the standard ingestion pipeline for acceptance.
"""
import json
import os
import logging
from pathlib import Path
from src.connectors.arxiv import ArxivConnector
from src.pipeline.ingestion import PaperIngestionPipeline
from src.storage.metadata_store import MetadataStore

logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_PATH = BASE_DIR / "data" / "metadata" / "papers.json"

# Load current papers
with open(PAPERS_PATH, "r", encoding="utf-8") as f:
    papers = json.load(f)

# IDs to remove
REMOVE_IDS = {"HC046", "HC047", "HC048"}

# Print what we're removing
print("=== Papers Being Removed ===")
for p in papers:
    if p["paper_id"] in REMOVE_IDS:
        print(f"  {p['paper_id']}: {p['title']}")
        # Delete PDF
        local_path = p.get("local_path", "")
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
            print(f"    -> Deleted PDF: {local_path}")

# Remove from papers list
papers = [p for p in papers if p["paper_id"] not in REMOVE_IDS]
print(f"\nPapers remaining after removal: {len(papers)}")

# Save cleaned papers.json
with open(PAPERS_PATH, "w", encoding="utf-8") as f:
    json.dump(papers, f, indent=2, ensure_ascii=False)
print("Saved cleaned papers.json")

# Instantiate MetadataStore and PaperIngestionPipeline
metadata_store = MetadataStore()
# Remove the old papers from metadata_store in-memory as well so it doesn't consider them duplicates
metadata_store.accepted_papers = [p for p in metadata_store.accepted_papers if p.paper_id not in REMOVE_IDS]

pipeline = PaperIngestionPipeline(metadata_store=metadata_store)

# Now discover and ingest replacements
print("\n=== Discovering Healthcare-Specific XAI Replacements ===")

arxiv = ArxivConnector()

# Best healthcare-specific queries
replacement_queries = [
    "explainable medical AI",
    "interpretable medical diagnosis",
    "interpretable clinical decision",
]

needed = 3
accepted = 0
next_id_num = 46  # HC046, HC047, HC048

# Get existing source_ids in metadata_store to avoid duplication
existing_source_ids = {p.source_id for p in metadata_store.accepted_papers if p.source_id}

for query in replacement_queries:
    if accepted >= needed:
        break
    print(f"\nQuery: '{query}'")
    candidates = arxiv.search_candidates(
        query=query, domain="healthcare",
        subtopic="explainable_ai_in_healthcare", max_results=10
    )
    
    for candidate in candidates:
        if accepted >= needed:
            break
        
        # Skip if already in dataset
        if candidate.source_id in existing_source_ids:
            print(f"  SKIP (duplicate): {candidate.title}")
            continue
        
        # Check title/abstract for healthcare relevance
        title_lower = candidate.title.lower()
        abstract_lower = (candidate.abstract or "").lower()
        healthcare_terms = ["medical", "clinical", "patient", "diagnosis", "healthcare",
                          "radiology", "pathology", "disease", "health", "biomedical",
                          "surgical", "cardiac", "cancer", "imaging", "physician", "icu", "mri", "ct", "ultrasound", "eeg"]
        
        is_healthcare = any(term in title_lower or term in abstract_lower for term in healthcare_terms)
        if not is_healthcare:
            print(f"  SKIP (not healthcare): {candidate.title}")
            continue
        
        # Use the pipeline to process this candidate
        print(f"  Processing: {candidate.title}")
        
        is_ok, metadata, msg = pipeline.process_candidate(candidate)
        if is_ok:
            accepted += 1
            next_id_num += 1
            existing_source_ids.add(candidate.source_id)
            print(f"  -> ACCEPTED as {metadata.paper_id}")
        else:
            print(f"  -> REJECTED: {msg}")

print(f"\n=== Summary ===")
print(f"Papers removed: {len(REMOVE_IDS)}")
print(f"Replacements accepted: {accepted}")

# Verify final count
with open(PAPERS_PATH, "r", encoding="utf-8") as f:
    final_papers = json.load(f)
hc_count = sum(1 for p in final_papers if p["domain"] == "healthcare")
print(f"Total Healthcare papers: {hc_count}")
print(f"Total papers in dataset: {len(final_papers)}")
