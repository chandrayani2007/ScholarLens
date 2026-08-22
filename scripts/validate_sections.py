import json
from collections import Counter, defaultdict

sections = []
with open("data/metadata/processed/sections.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        sections.append(json.loads(line))

# Check the spurious section names
print("=== Spurious section name samples ===")
for s in sections:
    if s["section_name"] == "Mariam Wehbe And Laurent Bobelin":
        sid = s["section_id"]
        pid = s["paper_id"]
        print(f"  {sid} in {pid} (order {s['section_order']}, pages {s['page_start']}-{s['page_end']})")
        break

for s in sections:
    if s["section_name"] == "Decision-Theoretic Optimization To Solve Tasks":
        sid = s["section_id"]
        pid = s["paper_id"]
        print(f"  {sid} in {pid} (order {s['section_order']}, pages {s['page_start']}-{s['page_end']})")
        break

for s in sections:
    if s["section_name"] == "Mse":
        sid = s["section_id"]
        pid = s["paper_id"]
        print(f"  {sid} in {pid} (order {s['section_order']}, pages {s['page_start']}-{s['page_end']})")
        break

# Check papers with 1 section
print("\n=== Papers with 1 section ===")
for s in sections:
    if s["paper_id"] == "AI019":
        print(f"  AI019: section={s['section_name']}, text_len={len(s['text'])}, pages={s['page_start']}-{s['page_end']}")
for s in sections:
    if s["paper_id"] == "AI032":
        print(f"  AI032: section={s['section_name']}, text_len={len(s['text'])}, pages={s['page_start']}-{s['page_end']}")
for s in sections:
    if s["paper_id"] == "HC040":
        print(f"  HC040: section={s['section_name']}, text_len={len(s['text'])}, pages={s['page_start']}-{s['page_end']}")

# Paper with max sections
spp = Counter(s["paper_id"] for s in sections)
max_paper = spp.most_common(1)[0]
print(f"\nPaper with most sections: {max_paper[0]} ({max_paper[1]} sections)")

# Section ID uniqueness
all_sec_ids = [s["section_id"] for s in sections]
print(f"\nTotal section IDs: {len(all_sec_ids)}")
print(f"Unique section IDs: {len(set(all_sec_ids))}")
print(f"Duplicate section IDs: {len(all_sec_ids) - len(set(all_sec_ids))}")

# Page ordering
bad_pages = [(s["section_id"], s["page_start"], s["page_end"]) for s in sections if s["page_start"] > s["page_end"]]
print(f"Sections with page_start > page_end: {len(bad_pages)}")

# Section order continuity per paper
paper_sections = defaultdict(list)
for s in sections:
    paper_sections[s["paper_id"]].append(s["section_order"])
gaps = 0
for pid, orders in paper_sections.items():
    expected = list(range(1, len(orders) + 1))
    if sorted(orders) != expected:
        gaps += 1
print(f"Papers with section order gaps: {gaps}")

# Verify subtopic linkage consistency
with open("data/metadata/papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)
paper_map = {p["paper_id"]: p for p in papers}

mismatches = []
for s in sections:
    p = paper_map.get(s["paper_id"])
    if p:
        if s["domain"] != p["domain"]:
            mismatches.append((s["section_id"], "domain", s["domain"], p["domain"]))
        if s["subtopic"] != p["subtopic"]:
            mismatches.append((s["section_id"], "subtopic", s["subtopic"], p["subtopic"]))
print(f"\nDomain/subtopic mismatches vs papers.json: {len(mismatches)}")

# File size
import os
fsize = os.path.getsize("data/metadata/processed/sections.jsonl")
print(f"\nsections.jsonl file size: {fsize / (1024*1024):.1f} MB")
