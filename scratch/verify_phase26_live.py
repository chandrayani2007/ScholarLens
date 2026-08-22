"""
Comprehensive Live API Verification Script for Phase 26
Verifies the 5 Mandatory Live Benchmark Queries against local FastAPI TestClient
"""

import sys
from pathlib import Path
sys.path.insert(0, r"c:\Users\nugur\Desktop\researchmind")

import json
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import get_current_user
from app.db.models import User

def override_get_current_user():
    return User(id=1, email="researcher@mind.edu", hashed_password="xxx", is_active=True)

app.dependency_overrides[get_current_user] = override_get_current_user
client = TestClient(app)

print("=== STARTING PHASE 26 LIVE API BENCHMARK VERIFICATION ===")

benchmark_queries = [
    {
        "q": "What are the limitations of retrieval-augmented generation systems?",
        "domain": "artificial_intelligence",
        "expected_domains": ["artificial_intelligence"],
        "intent": "Limitation"
    },
    {
        "q": "How does explainable AI improve trust in healthcare?",
        "domain": None,
        "expected_domains": ["artificial_intelligence", "healthcare"],
        "intent": "Effect"
    },
    {
        "q": "What are the challenges of using deep learning for cybersecurity?",
        "domain": "cybersecurity",
        "expected_domains": ["cybersecurity"],
        "intent": "Challenge"
    },
    {
        "q": "What are the benefits and limitations of smart irrigation systems?",
        "domain": "agriculture",
        "expected_domains": ["agriculture"],
        "intent": "Advantage"
    },
    {
        "q": "What are the major challenges in machine learning-based climate prediction?",
        "domain": "climate",
        "expected_domains": ["climate"],
        "intent": "Challenge"
    }
]

for idx, item in enumerate(benchmark_queries, 1):
    q = item["q"]
    dom = item["domain"]
    expected_doms = item["expected_domains"]

    payload = {"question": q, "top_k": 10}
    if dom:
        payload["domain"] = dom

    print(f"\n--- Benchmark Query #{idx}: '{q}' (Domain: {dom}) ---")
    res = client.post("/research/query", json=payload)
    assert res.status_code == 200, f"HTTP {res.status_code}: {res.text}"

    data = res.json()
    ans = data["answer"]
    conf = data["confidence"]
    why = data["why_this_answer"]
    cits = data["citations"]
    ev_list = data["evidence"]
    meta = data["retrieval_metadata"]

    print(f"  Answer Length: {len(ans.split())} words")
    print(f"  Answer Snippet: {ans[:140]}...")
    print(f"  Confidence: {conf} | Evidence Strength: {why['evidence_strength']}")
    print(f"  Domain Scope: {data['domain_scope']} | Why Scope: {why['domain_scope']}")
    print(f"  Bullets Count: {len(why['bullet_points'])}")
    for b in why['bullet_points']:
        print("    ", b.encode('ascii', 'ignore').decode('ascii'))

    # 1. Answer depth & non-repetition
    assert len(ans.split()) >= 40, f"Answer too short: {len(ans.split())} words"
    assert not ans.lower().startswith(q.lower().rstrip("?")), "Answer restates question!"
    assert "insufficient evidence" not in ans.lower()

    # 2. Confidence invariant
    assert conf == why["evidence_strength"], f"Confidence mismatch: {conf} vs {why['evidence_strength']}"

    # 3. Domain scope pre-filtering & non-leakage
    for dom_item in expected_doms:
        assert dom_item in data["domain_scope"], f"Missing expected domain '{dom_item}'"

    for ev in ev_list:
        assert ev["domain"] in data["domain_scope"], f"Domain leakage in evidence: '{ev['domain']}' not in {data['domain_scope']}"

    for tag, cit in cits.items():
        assert cit["citation_id"] in cits
        if cit.get("source_type") == "online":
            assert cit["title"] is not None
            assert cit["url"].startswith("http")

    print(f"  [PASSED] Benchmark Query #{idx} PASSED ALL CRITERIA!")

print("\n=== ALL 5 PHASE 26 LIVE API BENCHMARK VERIFICATION QUERIES PASSED 100%! ===")
