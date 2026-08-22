"""
Phase 27 Acceptance Fix — Live API Verification Script Printing Complete Uncut JSON Outputs
"""

import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\nugur\Desktop\researchmind")

import json
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import get_current_user
from app.db.models import User
from src.pipeline.rag import CitationValidator

def override_get_current_user():
    return User(id=1, email="researcher@mind.edu", hashed_password="xxx", is_active=True)

app.dependency_overrides[get_current_user] = override_get_current_user
client = TestClient(app)

print("==========================================================================")
print("=== PHASE 27 ACCEPTANCE FIX — LIVE API COMPLETE JSON VERIFICATION ===")
print("==========================================================================")

test_cases = [
    {
        "type": "TYPE A — LOCAL-ONLY QUERY",
        "q": "What are the limitations of retrieval-augmented generation systems?",
        "domain": "artificial_intelligence"
    },
    {
        "type": "TYPE B — AI CORPUS-SHORTAGE QUERY (Out-of-Corpus Trigger)",
        "q": "What is quantum teleportation using biological neural networks?",
        "domain": "artificial_intelligence"
    },
    {
        "type": "TYPE C — GENUINE ONLINE-ONLY QUERY (ArXiv Real Metadata Fallback)",
        "q": "What is surface code quantum error correction?",
        "domain": None
    },
    {
        "type": "TYPE D — LOCAL + ONLINE HYBRID QUERY",
        "q": "How do vector database indexes optimize retrieval in RAG systems?",
        "domain": "artificial_intelligence"
    },
    {
        "type": "TYPE E — AI + HEALTHCARE MULTI-DOMAIN QUERY",
        "q": "How is AI used in medical image analysis?",
        "domain": None
    }
]

for item in test_cases:
    t_type = item["type"]
    q = item["q"]
    dom = item["domain"]

    print(f"\n##########################################################################")
    print(f"# {t_type}")
    print(f"# Question: '{q}' (Selected Domain Filter: {dom})")
    print(f"##########################################################################")

    payload = {"question": q, "top_k": 10}
    if dom:
        payload["domain"] = dom

    res = client.post("/research/query", json=payload)
    assert res.status_code == 200, f"HTTP {res.status_code}: {res.text}"

    data = res.json()
    
    # Print complete formatted JSON response
    json_output = json.dumps(data, indent=2)
    print(json_output)

    # Acceptance Assertions
    ans = data["answer"]
    conf = data["confidence"]
    why = data["why_this_answer"]
    cits = data["citations"]
    ev_items = data["evidence"]

    tags_in_text = CitationValidator.extract_citations(ans)
    cit_keys = list(cits.keys())

    # 1. Answer completeness & non-repetition
    if "insufficient evidence" not in ans.lower():
        words_count = len(ans.split())
        assert words_count >= 90, f"Answer too short ({words_count} words < 90 words)"
        assert not ans.lower().startswith(q.lower().rstrip("?")), "Answer restates question!"
    
    # 2. Strict Confidence == Evidence Strength invariant
    assert conf == why["evidence_strength"], f"Confidence mismatch ({conf} vs {why['evidence_strength']})"

    # 3. 100% Bidirectional Citation Alignment: inline tags MUST match citations keys exactly
    for tag in tags_in_text:
        assert tag in cits, f"Inline tag [{tag}] present in answer but missing from citations dict!"
    for tag in cits.keys():
        assert tag in tags_in_text, f"Tag [{tag}] in citations dict but missing from answer text!"

    # 4. Real ArXiv Metadata Verification for Online Citations
    if "ONLINE-ONLY" in t_type or "SHORTAGE" in t_type:
        has_online_cits = any(t.startswith("O") for t in cits.keys())
        assert has_online_cits or data["confidence"] == "Insufficient"
        if has_online_cits:
            online_cits_list = [c for c in cits.values() if c.get("source_type") == "online"]
            for o_cit in online_cits_list:
                assert o_cit["url"].startswith("http://arxiv.org/abs/")
                assert o_cit["paper_id"].startswith("arXiv:")
                assert o_cit["title"] is not None
                assert len(o_cit["title"]) > 5

print("\n==========================================================================")
print("=== ALL 5 PHASE 27 ACCEPTANCE FIX LIVE JSON API QUERIES PASSED 100%! ===")
print("==========================================================================")
