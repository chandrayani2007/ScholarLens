import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

with open(BASE_DIR / "evaluation" / "test_queries.json", "r", encoding="utf-8") as f:
    queries = {q["query_id"]: q for q in json.load(f)}

with open(BASE_DIR / "evaluation" / "ground_truth.json", "r", encoding="utf-8") as f:
    gt = json.load(f)

with open(BASE_DIR / "data" / "indexes" / "chunk_embeddings_ids.json", "r", encoding="utf-8") as f:
    units_by_id = {u.get("unit_id", u["chunk_id"]): u for u in json.load(f)}

sample_qids = ["AI_Q001", "CY_Q001", "AG_Q001", "HC_Q001", "CL_Q001"]

print("=" * 75)
print("GROUND TRUTH MANUAL SANITY CHECK RESULTS")
print("=" * 75)

for qid in sample_qids:
    q = queries[qid]
    gt_entries = gt[qid]
    print(f"\n=== SANITY CHECK: {qid} ({q['domain']} / {q['subtopic']}) ===")
    print(f"Question: \"{q['question']}\"")
    print(f"Total Refined Ground Truth Entries: {len(gt_entries)}")
    for idx, item in enumerate(gt_entries[:3], 1):
        u = units_by_id.get(item["unit_id"])
        sec = u.get("section_name", "") if u else "N/A"
        txt_snippet = (u.get("text", "")[:120].replace("\n", " ") + "...") if u else "N/A"
        print(f"  GT Item {idx}: unit_id={item['unit_id']}, rel={item['relevance']}, section=\"{sec}\"")
        print(f"    Snippet: {txt_snippet}")
print("=" * 75)
