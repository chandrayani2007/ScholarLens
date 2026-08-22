import json
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "metadata" / "collection_log.json"

with open(LOG_PATH, "r", encoding="utf-8") as f:
    events = json.load(f)

# The three new papers
new_papers_info = [
    {
        "source_id": "2505.06898v1",
        "title": "Multi-Modal Explainable Medical AI Assistant for Trustworthy Human-AI Collaboration",
        "paper_id": "HC048"
    },
    {
        "source_id": "2512.18177v1",
        "title": "NEURO-GUARD: Neuro-Symbolic Generalization and Unbiased Adaptive Routing for Diagnostics -- Explainable Medical AI",
        "paper_id": "HC049"
    },
    {
        "source_id": "2001.05149v2",
        "title": "CheXplain: Enabling Physicians to Explore and UnderstandData-Driven, AI-Enabled Medical Imaging Analysis",
        "paper_id": "HC050"
    }
]

now_str = datetime.now(timezone.utc).isoformat()

for p in new_papers_info:
    # Check if a discovery event already exists
    exists = any(
        e.get("source_id") == p["source_id"] and e.get("action") == "candidate_discovered"
        for e in events
    )
    if not exists:
        ev = {
            "timestamp": now_str,
            "domain": "healthcare",
            "subtopic": "explainable_ai_in_healthcare",
            "source": "arXiv",
            "action": "candidate_discovered",
            "source_id": p["source_id"],
            "doi": None,
            "title": p["title"],
            "reason": None,
            "paper_id": p["paper_id"]
        }
        events.append(ev)
        print(f"Added discovery event for {p['paper_id']}")

with open(LOG_PATH, "w", encoding="utf-8") as f:
    json.dump(events, f, indent=2, ensure_ascii=False)
print("Saved collection_log.json")
