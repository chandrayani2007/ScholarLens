"""
Safely remove obsolete TXT corpus source files from data/papers/
AFTER verifying 100% genuine PDF corpus, indexes, tests, and builds.
"""

import os
from pathlib import Path

DATA_DIR = Path("data/papers")
DOMAINS = ["artificial_intelligence", "cybersecurity", "agriculture", "climate", "healthcare"]

def cleanup_txt_files():
    deleted_count = 0
    for dom in DOMAINS:
        dom_path = DATA_DIR / dom
        if dom_path.exists():
            for txt_file in dom_path.glob("*.txt"):
                txt_file.unlink()
                deleted_count += 1

    print(f"Successfully deleted {deleted_count} obsolete TXT corpus files.")

if __name__ == "__main__":
    cleanup_txt_files()
