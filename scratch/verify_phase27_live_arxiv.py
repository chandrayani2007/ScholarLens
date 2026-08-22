"""
Phase 27 — Real-Source Provenance End-to-End Verification Script
Executes live query against /research/query endpoint, prints complete JSON output,
and independently queries export.arxiv.org API to prove zero fabrication.
"""

import sys
import io
from pathlib import Path

# Ensure UTF-8 output encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, r"c:\Users\nugur\Desktop\researchmind")

import json
import xml.etree.ElementTree as ET
import urllib.request
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import get_current_user
from app.db.models import User

def override_get_current_user():
    return User(id=1, email="researcher@mind.edu", hashed_password="xxx", is_active=True)

app.dependency_overrides[get_current_user] = override_get_current_user
client = TestClient(app)

print("==========================================================================")
print("=== PHASE 27 REAL-SOURCE PROVENANCE — LIVE API VERIFICATION ===")
print("==========================================================================")

q = "What is surface code quantum error correction?"
print(f"\nSubmitting live API request: '{q}'...\n")

res = client.post("/research/query", json={"question": q, "top_k": 10})
assert res.status_code == 200, f"HTTP {res.status_code}: {res.text}"

data = res.json()

# Print complete, uncut JSON output
json_output = json.dumps(data, indent=2, ensure_ascii=False)
print(json_output)

print("\n--------------------------------------------------------------------------")
print("--- INDEPENDENT LIVE ARXIV SOURCE RESOLUTION VERIFICATION ---")
print("--------------------------------------------------------------------------")

citations = data.get("citations", {})
online_cits = {k: v for k, v in citations.items() if v.get("source_type") == "online"}

assert len(online_cits) > 0, "No online citations returned in live response!"

for cit_id, cit in online_cits.items():
    paper_id = cit["paper_id"]
    title = cit["title"]
    authors = cit["authors"]
    published = cit["published"]
    url = cit["url"]

    print(f"\n[VERIFYING {cit_id}] Paper ID: {paper_id}")
    print(f"  Title: {title}")
    print(f"  Authors: {authors}")
    print(f"  Published: {published}")
    print(f"  URL: {url}")

    assert paper_id.startswith("arXiv:"), f"Invalid ArXiv ID: {paper_id}"
    assert authors is not None and len(authors) > 0, f"Missing authors in {cit_id}"
    assert published is not None and len(published) >= 4, f"Missing published date in {cit_id}"
    assert url.startswith("http://arxiv.org/abs/"), f"Invalid ArXiv URL: {url}"

    clean_id = paper_id.replace("arXiv:", "")
    arxiv_api_url = f"http://export.arxiv.org/api/query?id_list={clean_id}"
    
    req = urllib.request.Request(arxiv_api_url, headers={"User-Agent": "ResearchMind-ProvenanceVerification/1.0"})
    with urllib.request.urlopen(req, timeout=10) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)
    
    assert entry is not None, f"FAIL: ArXiv ID '{paper_id}' was NOT found on live ArXiv API!"
    
    title_elem = entry.find("atom:title", ns)
    api_title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""
    
    assert api_title == title, f"FAIL: Title mismatch! (Response: '{title}' vs ArXiv API: '{api_title}')"
    print(f"  [OK] VERIFIED: {paper_id} exists on live ArXiv API and title matches perfectly!")

print("\n==========================================================================")
print("=== ALL ONLINE SOURCES VERIFIED AGAINST LIVE ARXIV API WITH 0 FABRICATION ===")
print("==========================================================================")
