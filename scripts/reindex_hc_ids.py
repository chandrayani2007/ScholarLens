import json
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_PATH = BASE_DIR / "data" / "metadata" / "papers.json"
LOG_PATH = BASE_DIR / "data" / "metadata" / "collection_log.json"

with open(PAPERS_PATH, "r", encoding="utf-8") as f:
    papers = json.load(f)

# Collect healthcare papers
hc_papers = [p for p in papers if p["domain"] == "healthcare"]
other_papers = [p for p in papers if p["domain"] != "healthcare"]

# Sort healthcare papers by current ID to ensure deterministic ordering
hc_papers.sort(key=lambda x: x["paper_id"])

print("Current mapping:")
id_map = {}
for i, p in enumerate(hc_papers):
    old_id = p["paper_id"]
    new_id = f"HC{i+1:03d}"
    id_map[old_id] = new_id
    print(f"  {old_id} -> {new_id}: {p['title']}")

# Perform renaming on files and update papers.json structure
for p in hc_papers:
    old_id = p["paper_id"]
    new_id = id_map[old_id]
    
    # Rename file if it exists and path matches
    old_path_str = p.get("local_path", "")
    if old_path_str:
        old_path = Path(old_path_str)
        # Construct new path string
        new_path = old_path.parent / f"{new_id}.pdf"
        
        # If old file exists, rename it (handling case where old_id == new_id)
        if old_path.exists() and old_id != new_id:
            if new_path.exists():
                os.remove(new_path) # clean up overwrite destination if exists
            os.rename(old_path, new_path)
            print(f"Renamed file: {old_path.name} -> {new_path.name}")
            
        p["local_path"] = str(new_path).replace("\\", "/")
    
    p["paper_id"] = new_id

# Update papers.json
all_updated_papers = other_papers + hc_papers
all_updated_papers.sort(key=lambda x: (x["domain"], x["paper_id"]))

with open(PAPERS_PATH, "w", encoding="utf-8") as f:
    json.dump(all_updated_papers, f, indent=2, ensure_ascii=False)
print("Saved reindexed papers.json")

# Update collection_log.json as a JSON document
if LOG_PATH.exists():
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        try:
            # Try to load as single JSON list
            events = json.load(f)
            is_json_lines = False
        except json.JSONDecodeError:
            # Fallback to JSON lines if it's not a single JSON doc
            is_json_lines = True
            
    if not is_json_lines:
        for event in events:
            if event.get("domain") == "healthcare":
                old_pid = event.get("paper_id")
                if old_pid in id_map:
                    event["paper_id"] = id_map[old_pid]
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        print("Updated collection_log.json (as JSON array)")
    else:
        # If it is JSON lines (one JSON per line)
        updated_events = []
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("domain") == "healthcare":
                        old_pid = event.get("paper_id")
                        if old_pid in id_map:
                            event["paper_id"] = id_map[old_pid]
                    updated_events.append(event)
                except Exception as e:
                    # Ignore comment or invalid lines
                    pass
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            for ev in updated_events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        print("Updated collection_log.json (as JSON lines)")
