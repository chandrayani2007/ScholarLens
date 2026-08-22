"""
Audit TXT dependencies across the entire repository.
Scans Python source files, config files, metadata, scripts, and tests for '.txt' usage.
"""

import os
import re
import json
from pathlib import Path

ROOT_DIR = Path(".")
IGNORE_DIRS = {".git", ".pytest_cache", "venv", "node_modules", "dist", ".gemini", "__pycache__"}


def scan_txt_dependencies():
    txt_references = []
    
    for path in ROOT_DIR.rglob("*"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        if path.is_file() and path.suffix in (".py", ".json", ".md", ".jsonl"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                lines = content.splitlines()
                for idx, line in enumerate(lines, 1):
                    if ".txt" in line.lower() or "txt" in line.lower():
                        if any(kw in line.lower() for kw in ["glob", "rglob", "txt", "file", "path", "open"]):
                            txt_references.append({
                                "file": str(path),
                                "line_num": idx,
                                "line": line.strip()
                            })
            except Exception:
                pass

    print(f"Found {len(txt_references)} TXT-related code/config lines across repository.")
    
    out_file = Path("evaluation/results/txt_dependency_audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(txt_references, f, indent=2)

    print(f"Saved dependency log to {out_file}")


if __name__ == "__main__":
    scan_txt_dependencies()
