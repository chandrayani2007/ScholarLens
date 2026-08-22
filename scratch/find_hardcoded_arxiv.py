import os
import re

search_dir = r"c:\Users\nugur\Desktop\researchmind"
matches = []

for root, dirs, files in os.walk(search_dir):
    if ".git" in root or "__pycache__" in root or ".pytest_cache" in root or "node_modules" in root:
        continue
    for file in files:
        if file.endswith(".py") or file.endswith(".js") or file.endswith(".ts") or file.endswith(".json"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "2401.05432" in content or "2311.09876" in content or "2401.12345" in content or "2403.99999" in content or "Fault-Tolerant Surface Code" in content:
                    matches.append((filepath, "Found hardcoded test string"))

print(f"Found {len(matches)} hardcoded occurrences:")
for path, msg in matches:
    print(f" - {path}: {msg}")
