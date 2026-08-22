import json
import fitz

with open("data/metadata/papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

for pid in ["HC046", "HC047", "HC048"]:
    p = next(x for x in papers if x["paper_id"] == pid)
    print(f"=== {pid} ===")
    print(f"Title: {p['title']}")
    print(f"Authors: {p['authors']}")
    print(f"Year: {p['publication_year']}")
    print(f"Subtopic: {p['subtopic']}")
    abstract = p.get("abstract", "N/A")
    print(f"Abstract: {abstract[:500]}")
    print()

    doc = fitz.open(p["local_path"])
    total_text = 0
    for i in range(doc.page_count):
        total_text += len(doc[i].get_text())
    print(f"Pages: {doc.page_count}, Total text chars: {total_text}")
    for i in range(min(5, doc.page_count)):
        t = doc[i].get_text()
        if len(t) > 50:
            print(f"Page {i+1} ({len(t)} chars): {t[:300]}...")
            break
    doc.close()
    print()
    print("---")
    print()
