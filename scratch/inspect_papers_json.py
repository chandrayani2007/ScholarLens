import json

with open(r"c:\Users\nugur\Desktop\researchmind\data\metadata\papers.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

print(f"Total papers in papers.json: {len(papers)}")

domain_counts = {}
for p in papers:
    dom = p.get("domain", "unknown")
    domain_counts[dom] = domain_counts.get(dom, 0) + 1

print("Domain counts:")
for dom, count in domain_counts.items():
    print(f"  {dom}: {count}")

if papers:
    print("\nSample paper structure:")
    sample = papers[0]
    for k, v in sample.items():
        if isinstance(v, str) and len(v) > 100:
            print(f"  {k}: {v[:100]}...")
        else:
            print(f"  {k}: {v}")
