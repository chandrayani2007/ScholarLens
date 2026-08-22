"""
Finish acquisition of missing Climate (CL133-CL200) and Healthcare (HC053-HC200) PDFs from ArXiv API using paginated queries.
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

CONFIGS = [
    {
        "domain": "climate",
        "code": "CL",
        "queries": [
            "cat:physics.ao-ph",
            "cat:physics.geo-ph",
            "all:climate+prediction",
            "all:climate+change",
            "all:meteorology",
            "all:global+warming"
        ],
        "start_idx": 51,
        "end_idx": 200
    },
    {
        "domain": "healthcare",
        "code": "HC",
        "queries": [
            "cat:q-bio.NC",
            "cat:q-bio.QM",
            "cat:eess.IV",
            "all:clinical",
            "all:healthcare",
            "all:medical+imaging",
            "all:electronic+health+records"
        ],
        "start_idx": 51,
        "end_idx": 200
    }
]


def fetch_arxiv_paginated(queries, target_needed=200):
    candidates = []
    seen_ids = set()
    
    for q in queries:
        for start in [0, 100, 200]:
            if len(candidates) >= target_needed + 50:
                break
            url = f"http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&start={start}&max_results=100&sortBy=submittedDate&sortOrder=descending"
            headers = {"User-Agent": "ScholarLens-CorpusCollector/1.0"}
            req = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    xml_data = resp.read()
                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
                
                for entry in root.findall("atom:entry", ns):
                    arxiv_id_elem = entry.find("atom:id", ns)
                    title_elem = entry.find("atom:title", ns)
                    summary_elem = entry.find("atom:summary", ns)
                    published_elem = entry.find("atom:published", ns)
                    
                    if arxiv_id_elem is None or title_elem is None:
                        continue
                    raw_id = arxiv_id_elem.text.strip().split("/abs/")[-1]
                    if raw_id in seen_ids:
                        continue
                    seen_ids.add(raw_id)

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
                    
                    candidates.append({
                        "arxiv_id": f"arXiv:{raw_id}",
                        "raw_id": raw_id,
                        "title": title,
                        "authors": authors,
                        "published_date": published,
                        "abstract": summary,
                        "pdf_url": pdf_url,
                        "abs_url": abs_url
                    })
                time.sleep(0.5)
            except Exception as e:
                print(f"Error fetching {q} (start={start}): {e}")
                
    return candidates


def finish_acquisition():
    downloader = PdfDownloader()
    validator = PdfValidator(min_text_length=500)
    
    existing_meta = {}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                existing_meta = json.load(f)
        except Exception:
            existing_meta = {}

    for cfg in CONFIGS:
        dom_name = cfg["domain"]
        code = cfg["code"]
        start_idx = cfg["start_idx"]
        end_idx = cfg["end_idx"]
        dom_dir = DATA_DIR / dom_name
        dom_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFetching candidates for {dom_name}...")
        candidates = fetch_arxiv_paginated(cfg["queries"], target_needed=150)
        print(f"Acquired {len(candidates)} candidate papers for {dom_name}.")

        cand_idx = 0
        for num in range(start_idx, end_idx + 1):
            paper_id = f"{code}{num:03d}"
            pdf_target_path = dom_dir / f"{paper_id}.pdf"

            if pdf_target_path.exists():
                val_res = validator.validate(pdf_target_path)
                if val_res.is_valid:
                    continue

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
                        if pdf_target_path.exists():
                            pdf_target_path.unlink()
                        os.replace(str(temp_pdf), str(pdf_target_path))

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

                        download_success = True
                        time.sleep(0.5)
                        break
                    else:
                        print(f"    Rejected {pdf_url}: {val_res.error_reason}")
                        downloader.cleanup_temp_file(temp_pdf)
                except Exception as e:
                    print(f"    Failed downloading {pdf_url}: {e}")

            if not download_success:
                print(f"WARNING: Could not acquire PDF for {paper_id}!")

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_meta, f, indent=2)

    print("\nFinish acquisition completed.")


if __name__ == "__main__":
    finish_acquisition()
