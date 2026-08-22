"""
Phase 28.7 — Acquisition Script for 750 Genuine Open-Access PDFs
Downloads authentic research PDFs from ArXiv API for missing papers:
- Artificial Intelligence: AI051.pdf to AI200.pdf (150 papers)
- Cybersecurity: CY051.pdf to CY200.pdf (150 papers)
- Agriculture: AG051.pdf to AG200.pdf (150 papers)
- Climate: CL051.pdf to CL200.pdf (150 papers)
- Healthcare: HC051.pdf to HC200.pdf (150 papers)

Maintains full metadata provenance (title, authors, published date, arXiv ID, PDF URL).
Validates each downloaded PDF using PyMuPDF to ensure it is a valid full-text research paper.
"""

import json
import os
import re
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from src.downloader.pdf_downloader import PdfDownloader
from src.validator.pdf_validator import PdfValidator

DATA_DIR = Path("data/papers")
METADATA_FILE = Path("data/metadata/papers_metadata.json")

DOMAINS_CONFIG = [
    {
        "domain": "artificial_intelligence",
        "code": "AI",
        "query": "cat:cs.AI OR cat:cs.LG OR cat:cs.CL OR cat:cs.CV",
        "start_idx": 51,
        "end_idx": 200,
    },
    {
        "domain": "cybersecurity",
        "code": "CY",
        "query": "cat:cs.CR OR all:cybersecurity OR all:intrusion",
        "start_idx": 51,
        "end_idx": 200,
    },
    {
        "domain": "agriculture",
        "code": "AG",
        "query": "all:agriculture OR all:crop OR all:irrigation OR all:agronomy OR all:soil",
        "start_idx": 51,
        "end_idx": 200,
    },
    {
        "domain": "climate",
        "code": "CL",
        "query": "cat:physics.ao-ph OR cat:physics.geo-ph OR all:climate OR all:meteorology OR all:atmosphere OR all:warming",
        "start_idx": 51,
        "end_idx": 200,
    },
    {
        "domain": "healthcare",
        "code": "HC",
        "query": "cat:q-bio.NC OR cat:q-bio.QM OR cat:eess.IV OR all:clinical OR all:medical OR all:healthcare OR all:disease OR all:hospital",
        "start_idx": 51,
        "end_idx": 200,
    },
]


def fetch_arxiv_candidates(search_query: str, max_results: int = 350):
    url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(search_query)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    headers = {"User-Agent": "ScholarLens-CorpusCollector/1.0"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_data = resp.read()
        
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        
        entries = []
        for entry in root.findall("atom:entry", ns):
            arxiv_id_elem = entry.find("atom:id", ns)
            title_elem = entry.find("atom:title", ns)
            summary_elem = entry.find("atom:summary", ns)
            published_elem = entry.find("atom:published", ns)
            
            if arxiv_id_elem is None or title_elem is None:
                continue
                
            raw_id = arxiv_id_elem.text.strip().split("/abs/")[-1]
            title = re.sub(r"\s+", " ", title_elem.text.strip())
            summary = re.sub(r"\s+", " ", summary_elem.text.strip()) if summary_elem is not None else ""
            published = published_elem.text.strip()[:10] if published_elem is not None else ""
            
            authors = []
            for auth in entry.findall("atom:author", ns):
                name_elem = auth.find("atom:name", ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())

            pdf_url = f"http://arxiv.org/pdf/{raw_id}.pdf"
            abs_url = f"http://arxiv.org/abs/{raw_id}"
            
            entries.append({
                "arxiv_id": f"arXiv:{raw_id}",
                "raw_id": raw_id,
                "title": title,
                "authors": authors,
                "published_date": published,
                "abstract": summary,
                "pdf_url": pdf_url,
                "abs_url": abs_url
            })
            
        return entries
    except Exception as e:
        print(f"Error fetching ArXiv for query '{search_query}': {e}")
        return []


def run_acquisition():
    downloader = PdfDownloader()
    validator = PdfValidator(min_text_length=500)
    
    # Load existing metadata
    existing_meta = {}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    existing_meta = data
                elif isinstance(data, list):
                    existing_meta = {item["paper_id"]: item for item in data if "paper_id" in item}
        except Exception:
            existing_meta = {}

    acquired_count = 0
    failed_count = 0

    print("==========================================================================")
    print("STARTING ACQUISITION OF 750 GENUINE RESEARCH PAPERS (PDFs)")
    print("==========================================================================")

    for cfg in DOMAINS_CONFIG:
        dom_name = cfg["domain"]
        code = cfg["code"]
        query = cfg["query"]
        start_idx = cfg["start_idx"]
        end_idx = cfg["end_idx"]

        dom_dir = DATA_DIR / dom_name
        dom_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFetching candidates for {dom_name} [{code}051 - {code}200]...")
        candidates = fetch_arxiv_candidates(query, max_results=350)
        print(f"Found {len(candidates)} candidate papers from ArXiv.")

        cand_idx = 0
        for num in range(start_idx, end_idx + 1):
            paper_id = f"{code}{num:03d}"
            pdf_target_path = dom_dir / f"{paper_id}.pdf"

            # Check if genuine PDF already exists and is valid
            if pdf_target_path.exists():
                val_res = validator.validate(pdf_target_path)
                if val_res.is_valid:
                    print(f"  * {paper_id}.pdf already exists and is valid.")
                    continue

            # Find next valid candidate
            download_success = False
            while cand_idx < len(candidates):
                cand = candidates[cand_idx]
                cand_idx += 1

                pdf_url = cand["pdf_url"]
                safe_title = cand['title'][:40].encode('ascii', 'ignore').decode('ascii')
                print(f"  * Downloading {paper_id}.pdf from {pdf_url} ('{safe_title}')...")

                try:
                    temp_pdf = downloader.download_temp_pdf(pdf_url, DATA_DIR / "temp")
                    val_res = validator.validate(temp_pdf)

                    if val_res.is_valid and val_res.text_length >= 500:
                        # Move valid PDF into place
                        if pdf_target_path.exists():
                            pdf_target_path.unlink()
                        os.replace(str(temp_pdf), str(pdf_target_path))

                        # Save metadata
                        existing_meta[paper_id] = {
                            "paper_id": paper_id,
                            "title": cand["title"],
                            "authors": cand["authors"],
                            "published_date": cand["published_date"],
                            "domain": dom_name,
                            "file_path": f"data/papers/{dom_name}/{paper_id}.pdf",
                            "source_type": "pdf",
                            "url": cand["abs_url"],
                            "arxiv_id": cand["arxiv_id"],
                            "abstract": cand["abstract"]
                        }

                        acquired_count += 1
                        download_success = True
                        time.sleep(0.5)  # Respect ArXiv rate limits
                        break
                    else:
                        print(f"    Rejected {pdf_url}: {val_res.error_reason} (text len={val_res.text_length})")
                        downloader.cleanup_temp_file(temp_pdf)
                except Exception as e:
                    print(f"    Failed downloading {pdf_url}: {e}")

            if not download_success:
                print(f"WARNING: Could not acquire genuine PDF for {paper_id}!")
                failed_count += 1

    # Save updated metadata
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_meta, f, indent=2)

    print("\n==========================================================================")
    print(f"ACQUISITION COMPLETE. Newly acquired: {acquired_count} PDFs. Failed: {failed_count}")
    print("==========================================================================")


if __name__ == "__main__":
    run_acquisition()
