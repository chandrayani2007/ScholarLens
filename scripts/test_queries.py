import json
import time
from pathlib import Path
from src.connectors.arxiv import ArxivConnector

def improve_query(query: str) -> str:
    """Fallback mechanism if query returns 0. Keep only the first word or two."""
    parts = query.split()
    if len(parts) > 2:
        return " ".join(parts[:2])
    elif len(parts) == 2:
        return parts[0]
    return query

def main():
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / "data" / "metadata" / "new_queries.json", "r") as f:
        new_queries = json.load(f)
        
    with open(base_dir / "config" / "domains.json", "r") as f:
        old_config = json.load(f)["domains"]
        
    arxiv = ArxivConnector()
    
    report = []
    updated_domains = {}

    for domain, subtopics in new_queries.items():
        print(f"Testing domain: {domain}")
        updated_domains[domain] = {}
        report.append(f"### Domain: {domain.replace('_', ' ').title()}\n")
        report.append("| Subtopic | Old Queries | New Queries | Discovered | Source | Status | Rationale |")
        report.append("|---|---|---|---|---|---|---|")
        
        # Get old queries mapping
        old_queries_map = {}
        for sub_cfg in old_config.get(domain, {}).get("subtopics", []):
            old_queries_map[sub_cfg["canonical"]] = sub_cfg["search_queries"]

        for subtopic, queries in subtopics.items():
            print(f"  Testing subtopic: {subtopic}")
            old_qs = old_queries_map.get(subtopic, [])
            final_queries = []
            
            for query in queries:
                q = query
                max_attempts = 3
                for attempt in range(max_attempts):
                    # We use max_results=5 to make it fast and just check if >0
                    candidates = arxiv.search_candidates(query=q, domain=domain, subtopic=subtopic, max_results=5)
                    count = len(candidates)
                    if count > 0:
                        final_queries.append(q)
                        report.append(f"| {subtopic} | `{old_qs}` | `{q}` | {count} | arXiv | Acceptable | Broader keywords, replacing long exact phrases. |")
                        break
                    else:
                        print(f"    [!] Query '{q}' returned 0. Improving...")
                        next_q = improve_query(q)
                        if next_q == q:
                            final_queries.append(q)
                            report.append(f"| {subtopic} | `{old_qs}` | `{q}` | 0 | arXiv | Unacceptable | Could not improve further. |")
                            break
                        q = next_q
                        time.sleep(3) # Wait before retry
                time.sleep(3) # Polite delay between normal queries
                
            updated_domains[domain][subtopic] = final_queries
            
    # Write the report
    with open(base_dir / "data" / "metadata" / "query_test_report.md", "w") as f:
        f.write("\n".join(report))
        
    # Write the updated structure
    with open(base_dir / "data" / "metadata" / "updated_domains.json", "w") as f:
        json.dump(updated_domains, f, indent=2)

if __name__ == "__main__":
    main()
