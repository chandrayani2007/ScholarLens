import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from src.config.loader import load_domain_config
from src.discovery.service import CandidateDiscoveryService
from src.pipeline.ingestion import PaperIngestionPipeline
from src.storage.metadata_store import MetadataStore

logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.collect_corpus <domain_name>")
        sys.exit(1)
        
    domain_name = sys.argv[1]
    
    config = load_domain_config()
    discovery = CandidateDiscoveryService(config=config)
    metadata_store = MetadataStore()
    pipeline = PaperIngestionPipeline(metadata_store=metadata_store)

    domain_cfg = config.domains[domain_name]

    # Track accepted counts per subtopic
    accepted_counts = defaultdict(int)
    for paper in metadata_store.accepted_papers:
        if paper.domain == domain_name and paper.status == "downloaded":
            accepted_counts[paper.subtopic] += 1
    
    total_accepted = sum(accepted_counts.values())
    target_total = domain_cfg.target_count

    print(f"Current accepted {domain_name} papers: {total_accepted}/{target_total}")

    # Track stats for the run
    stats = {
        "run_accepted": 0,
        "run_rejected": 0,
        "run_duplicates": 0,
        "failures": []
    }

    def process_candidates(candidates_list, max_per_subtopic=None):
        nonlocal total_accepted
        for subtopic, candidates in candidates_list.items():
            if total_accepted >= target_total:
                break
            
            for cand in candidates:
                if total_accepted >= target_total:
                    break
                
                if max_per_subtopic and accepted_counts[subtopic] >= max_per_subtopic:
                    break
                
                print(f"Processing Candidate: '{cand.title}'")
                accepted, metadata, msg = pipeline.process_candidate(cand)
                
                if accepted:
                    accepted_counts[subtopic] += 1
                    total_accepted += 1
                    stats["run_accepted"] += 1
                    print(f" -> ACCEPTED: {metadata.paper_id} ({accepted_counts[subtopic]} for {subtopic})")
                else:
                    if "Duplicate detected" in msg:
                        stats["run_duplicates"] += 1
                    else:
                        stats["run_rejected"] += 1
                        stats["failures"].append(msg)
                    print(f" -> REJECTED/SKIPPED: {msg}")

    # Pass 1: Try to get exactly the target per subtopic (5)
    sources = ["arxiv"]
    
    for subtopic in domain_cfg.subtopics:
        if total_accepted >= target_total:
            break
        
        current_subtopic = accepted_counts[subtopic.canonical]
        target_subtopic = subtopic.target_count
        
        if current_subtopic >= target_subtopic:
            continue
            
        print(f"\n--- Discovering Subtopic: {subtopic.name} ---")
        candidates = discovery.discover_candidates_for_subtopic(
            domain_name=domain_name,
            subtopic_canonical=subtopic.canonical,
            source_names=sources,
            max_results_per_query=10
        )
        process_candidates({subtopic.canonical: candidates}, max_per_subtopic=target_subtopic)
        
    print("\n--- Pass 2: Fill remaining gaps ---")
    if total_accepted < target_total:
        for subtopic in domain_cfg.subtopics:
            if total_accepted >= target_total:
                break
            print(f"--- Filling gaps from {subtopic.name} ---")
            candidates = discovery.discover_candidates_for_subtopic(
                domain_name=domain_name,
                subtopic_canonical=subtopic.canonical,
                source_names=sources,
                max_results_per_query=15
            )
            process_candidates({subtopic.canonical: candidates}, max_per_subtopic=None)
        
    print("\n--- Pass 3: Expand search if still short ---")
    if total_accepted < target_total:
        sources = ["arxiv", "semantic_scholar"]
        for subtopic in domain_cfg.subtopics:
            if total_accepted >= target_total:
                break
            print(f"Expanding search for {subtopic.name}...")
            extra_candidates = discovery.discover_candidates_for_subtopic(
                domain_name=domain_name,
                subtopic_canonical=subtopic.canonical,
                source_names=sources,
                max_results_per_query=15
            )
            process_candidates({subtopic.canonical: extra_candidates}, max_per_subtopic=None)

    # ---------------------------------------------------------
    # GENERATE REPORT
    # ---------------------------------------------------------
    metadata_store.reload()
    ai_papers = [p for p in metadata_store.accepted_papers if p.domain == domain_name and p.status == "downloaded"]
    
    # 1. Number of accepted papers
    print(f"\n1. Number of accepted papers: {len(ai_papers)}")
    
    # 2. Number rejected
    print(f"2. Number rejected (this run): {stats['run_rejected']}")
    
    # 3. Number of duplicates
    print(f"3. Number of duplicates (this run): {stats['run_duplicates']}")
    
    # 4. Subtopic distribution
    dist = defaultdict(int)
    for p in ai_papers: dist[p.subtopic] += 1
    print("4. Subtopic distribution:")
    for k, v in dist.items(): print(f"   - {k}: {v}")
        
    # 5. Publication-year distribution
    years = defaultdict(int)
    for p in ai_papers: years[p.publication_year] += 1
    print("5. Publication-year distribution:")
    for k, v in years.items(): print(f"   - {k}: {v}")
        
    # 6. Source distribution
    sources_dist = defaultdict(int)
    for p in ai_papers: sources_dist[p.source] += 1
    print("6. Source distribution:")
    for k, v in sources_dist.items(): print(f"   - {k}: {v}")
        
    # 7. Average PDF size
    sizes = []
    base_dir = Path(__file__).resolve().parent.parent
    for p in ai_papers:
        path = base_dir / p.local_path
        if path.exists():
            sizes.append(path.stat().st_size)
    if sizes:
        avg_mb = (sum(sizes) / len(sizes)) / (1024 * 1024)
        print(f"7. Average PDF size: {avg_mb:.2f} MB")
    else:
        print("7. Average PDF size: 0 MB")
        
    # 8. Metadata completeness
    # All required fields are present if it's in pydantic model, check optional fields
    full_count = sum(1 for p in ai_papers if p.abstract and p.doi and p.venue)
    print(f"8. Metadata completeness (papers with abstract, doi, and venue): {full_count}/{len(ai_papers)}")
    
    # 9. Missing DOI count
    missing_doi = sum(1 for p in ai_papers if not p.doi)
    print(f"9. Missing DOI count: {missing_doi}")
    
    # 10. Missing abstract count
    missing_abstract = sum(1 for p in ai_papers if not p.abstract)
    print(f"10. Missing abstract count: {missing_abstract}")
    
    # 11. Collection failures
    print(f"11. Collection failures (first 5 unique): {list(set(stats['failures']))[:5]}")
    
    # 12. Final papers.json statistics
    print(f"12. Final papers.json total records: {len(metadata_store.accepted_papers)}")
    
    # 13. collection_log.json statistics
    log_path = base_dir / "data" / "metadata" / "collection_log.json"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        print(f"13. collection_log.json events: {len(log_data)}")
    else:
        print("13. collection_log.json not found")
        
    # 14. Final AI paper IDs
    ids = sorted([p.paper_id for p in ai_papers])
    print(f"14. Final AI paper IDs: {ids[0]} to {ids[-1]}")
    
    # 15. Confirmation
    print(f"15. Confirmation that exactly 50 accepted AI papers exist: {len(ai_papers) == 50}")

if __name__ == "__main__":
    main()
