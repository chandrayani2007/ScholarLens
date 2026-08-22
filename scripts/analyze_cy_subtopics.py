import json
from collections import defaultdict
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    log_file = base_dir / "data" / "metadata" / "collection_log.json"
    
    with open(log_file, "r", encoding="utf-8") as f:
        logs = json.load(f)
        
    discovered = defaultdict(int)
    rejected = defaultdict(lambda: defaultdict(int))
    accepted = defaultdict(int)
    
    for event in logs:
        if event.get("domain") != "cybersecurity":
            continue
            
        sub = event.get("subtopic")
        action = event.get("action")
        
        if action == "candidate_discovered":
            discovered[sub] += 1
        elif action in ("rejected", "duplicate_detected", "download_failed", "validation_failed"):
            reason = event.get("reason", "Unknown")
            rejected[sub][reason] += 1
        elif action == "accepted":
            accepted[sub] += 1
            
    print("--- Discovered ---")
    for k, v in discovered.items():
        print(f"{k}: {v}")
        
    print("\n--- Rejected ---")
    for sub, reasons in rejected.items():
        print(f"{sub}:")
        for r, count in reasons.items():
            print(f"  {r}: {count}")
            
    print("\n--- Accepted ---")
    for k, v in accepted.items():
        print(f"{k}: {v}")
        
if __name__ == "__main__":
    main()
