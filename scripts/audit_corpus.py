import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
import fitz  # PyMuPDF

from src.config.loader import load_domain_config
from src.storage.metadata_store import MetadataStore

logging.basicConfig(level=logging.ERROR)

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.audit_corpus <domain_name>")
        sys.exit(1)
        
    domain_name = sys.argv[1]

    base_dir = Path(__file__).resolve().parent.parent
    config = load_domain_config()
    domain_subtopics = [s.canonical for s in config.domains[domain_name].subtopics]
    
    metadata_store = MetadataStore()
    papers = [p for p in metadata_store.accepted_papers if p.domain == domain_name and p.status == "downloaded"]
    
    report = []
    def add(text):
        report.append(text)
        print(text)

    add(f"# {domain_name.title()} Dataset Audit Report\n")
    
    # 1. Exactly 50 unique accepted papers
    add(f"## 1. Paper Count")
    add(f"Total {domain_name} papers in papers.json: {len(papers)}")
    if len(papers) == 50:
        add("[PASS] Exactly 50 papers exist.")
    else:
        add(f"[FAIL] Expected 50, found {len(papers)}.")
        
    # 2. No duplicates
    add(f"\n## 2. Duplicate Check")
    ids = [p.paper_id for p in papers]
    dois = [p.doi for p in papers if p.doi]
    arxiv_ids = [p.arxiv_id for p in papers if p.arxiv_id]
    source_ids = [p.source_id for p in papers if p.source_id]
    hashes = [p.file_hash for p in papers if p.file_hash]
    
    def check_dup(name, lst):
        dups = len(lst) - len(set(lst))
        if dups == 0:
            add(f"[PASS] No duplicate {name} values.")
        else:
            add(f"[FAIL] Found {dups} duplicate {name} values!")
            
    check_dup("paper_ids", ids)
    check_dup("DOIs", dois)
    check_dup("arXiv IDs", arxiv_ids)
    check_dup("source IDs", source_ids)
    check_dup("SHA-256 hashes", hashes)
    
    # 3. PDF Validity
    add(f"\n## 3. PDF Validity")
    invalid_pdfs = []
    total_pages = 0
    total_size = 0
    for p in papers:
        pdf_path = base_dir / p.local_path
        if not pdf_path.exists():
            invalid_pdfs.append(f"{p.paper_id}: File not found at {p.local_path}")
            continue
        
        size = pdf_path.stat().st_size
        if size == 0:
            invalid_pdfs.append(f"{p.paper_id}: Zero size file")
            continue
            
        total_size += size
        try:
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                invalid_pdfs.append(f"{p.paper_id}: Zero pages")
            else:
                total_pages += doc.page_count
                text = doc[0].get_text()
                if len(text.strip()) < 50:
                    invalid_pdfs.append(f"{p.paper_id}: Unreadable/non-extractable text on page 1")
            doc.close()
        except Exception as e:
            invalid_pdfs.append(f"{p.paper_id}: fitz error {e}")
            
    if not invalid_pdfs:
        add("[PASS] All PDFs are valid, readable, have extractable text, and non-zero size.")
    else:
        add("[FAIL] PDF Validation Failures:")
        for inv in invalid_pdfs:
            add(f"  - {inv}")
            
    # 4. Metadata Completeness
    add(f"\n## 4. Metadata Completeness")
    fields = ['title', 'authors', 'abstract', 'publication_year', 'source', 'doi', 'venue', 'pdf_url', 'source_url']
    completeness = {f: 0 for f in fields}
    for p in papers:
        if p.title: completeness['title'] += 1
        if p.authors and len(p.authors) > 0: completeness['authors'] += 1
        if p.abstract: completeness['abstract'] += 1
        if p.publication_year: completeness['publication_year'] += 1
        if p.source: completeness['source'] += 1
        if p.doi: completeness['doi'] += 1
        if p.venue: completeness['venue'] += 1
        if p.pdf_url: completeness['pdf_url'] += 1
        if p.source_url: completeness['source_url'] += 1
        
    for f, c in completeness.items():
        add(f"- {f}: {c}/{len(papers)} ({c/len(papers)*100:.1f}%)")
        
    # 5. Correct Subtopic
    add(f"\n## 5. Subtopic Validation")
    invalid_subtopics = [p.paper_id for p in papers if p.subtopic not in domain_subtopics]
    if not invalid_subtopics:
        add(f"[PASS] All papers belong to valid {domain_name} subtopics.")
    else:
        add(f"[FAIL] Invalid subtopics found in: {invalid_subtopics}")
        
    # 6. Show the 5 papers per subtopic
    add(f"\n## 6. Subtopic Papers")
    sub_papers = defaultdict(list)
    for p in papers:
        sub_papers[p.subtopic].append(p)
    
    for sub, lst in sub_papers.items():
        add(f"**{sub} ({len(lst)} papers):**")
        for p in lst:
            add(f"  - {p.paper_id}: {p.title}")
            
    # 7. Sequential IDs
    add(f"\n## 7. Sequential IDs")
    prefix = config.domains[domain_name].id_prefix
    expected_ids = [f"{prefix}{i:03d}" for i in range(1, 51)]
    actual_ids = sorted([p.paper_id for p in papers])
    if actual_ids == expected_ids:
        add(f"[PASS] IDs are strictly sequential from {prefix}001 to {prefix}050 with no gaps.")
    else:
        add("[FAIL] ID gaps or mismatches found.")
        add(f"Expected: {expected_ids}")
        add(f"Actual:   {actual_ids}")
        
    # 8. Statistics
    add(f"\n## 8. Corpus Statistics")
    avg_pages = total_pages / len(papers) if papers else 0
    avg_size_mb = (total_size / len(papers)) / (1024*1024) if papers else 0
    avg_abs_len = sum(len(p.abstract) for p in papers if p.abstract) / len(papers) if papers else 0
    avg_authors = sum(len(p.authors) for p in papers if p.authors) / len(papers) if papers else 0
    
    add(f"- Average pages: {avg_pages:.1f}")
    add(f"- Average PDF size: {avg_size_mb:.2f} MB")
    add(f"- Average abstract length (chars): {avg_abs_len:.1f}")
    add(f"- Average number of authors: {avg_authors:.1f}")
    
    # 9. Publication Year Histogram
    add(f"\n## 9. Publication-Year Histogram")
    years = defaultdict(int)
    for p in papers:
        years[p.publication_year] += 1
    for y in sorted(years.keys()):
        add(f"- {y}: {'*' * years[y]} ({years[y]})")
        
    # 10. Source Summary
    add(f"\n## 10. Source Summary")
    sources = defaultdict(int)
    for p in papers:
        sources[p.source] += 1
    for s, c in sources.items():
        add(f"- {s}: {c}")
        
    # 11. Collection Log Consistency
    add(f"\n## 11. collection_log.json Consistency")
    log_path = base_dir / "data" / "metadata" / "collection_log.json"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
        log_ids = set([lg.get('source_id') for lg in logs if lg.get('action') == 'candidate_discovered'])
        found_all = True
        for p in papers:
            if p.source_id not in log_ids:
                add(f"[FAIL] Missing log for source_id: {p.source_id}")
                found_all = False
        if found_all:
            add("[PASS] All accepted source_ids are present in collection_log.json.")
    else:
        add("[FAIL] collection_log.json not found.")
        
    # 12. Random Sample
    add(f"\n## 12. Random Inspection (5 Papers)")
    samples = random.sample(papers, 5) if len(papers) >= 5 else papers
    for i, p in enumerate(samples, 1):
        doc = fitz.open(base_dir / p.local_path)
        pages = doc.page_count
        doc.close()
        add(f"**Sample {i} - {p.paper_id}**")
        add(f"- Title: {p.title}")
        add(f"- Authors: {', '.join(p.authors)}")
        add(f"- DOI: {p.doi}")
        add(f"- Source: {p.source}")
        add(f"- Pages: {pages}")
        add(f"- PDF Path: {p.local_path}")
        add(f"- Abstract: {p.abstract[:200]}...")
        add("")
        
if __name__ == "__main__":
    main()
