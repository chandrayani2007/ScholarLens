import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import fitz

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_PATH = BASE_DIR / "data" / "metadata" / "papers.json"
LOG_PATH = BASE_DIR / "data" / "metadata" / "collection_log.json"

with open(PAPERS_PATH, "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Total Papers: {len(papers)}")

# Domain counts
domain_counts = Counter(p["domain"] for p in papers)
print("Domains:", dict(domain_counts))

# Subtopic counts
subtopic_counts = Counter(p["subtopic"] for p in papers)
print("Subtopics count:", len(subtopic_counts))

# Years
years = Counter(p["publication_year"] for p in papers)
print("Years:", dict(sorted(years.items())))

# Sources
sources = Counter(p["source"] for p in papers)
print("Sources:", dict(sources))

# Size and pages
total_size = 0
total_pages = 0
for p in papers:
    local_path = BASE_DIR / p["local_path"]
    if local_path.exists():
        total_size += local_path.stat().st_size
        try:
            doc = fitz.open(local_path)
            total_pages += doc.page_count
            doc.close()
        except:
            pass

print(f"Total Size: {total_size / (1024*1024):.2f} MB")
print(f"Average PDF Size: {total_size / len(papers) / (1024*1024):.2f} MB")
print(f"Total Pages: {total_pages}")
print(f"Average Pages: {total_pages / len(papers):.1f}")

# Metadata completeness
fields = ["title", "authors", "abstract", "publication_year", "doi", "venue", "source", "source_url", "pdf_url"]
completeness = {f: 0 for f in fields}
for p in papers:
    for f in fields:
        if p.get(f) is not None and str(p.get(f)).strip() != "" and p.get(f) != "None":
            completeness[f] += 1

print("\nCompleteness:")
for f, c in completeness.items():
    print(f"  {f}: {c}/{len(papers)} ({c/len(papers)*100:.1f}%)")

# Log stats
if LOG_PATH.exists():
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        logs = json.load(f)
    print(f"\nTotal Log Events: {len(logs)}")
    actions = Counter(l.get("action") for l in logs)
    print("Log Actions:", dict(actions))
