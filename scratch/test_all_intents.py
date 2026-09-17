import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def get_token():
    res = requests.post(f"{BASE_URL}/auth/login", data={"username": "testuser", "password": "password123"})
    if res.status_code == 200:
        return res.json()["access_token"]
    res = requests.post(f"{BASE_URL}/auth/register", json={
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "password123",
        "full_name": "Test User"
    })
    res = requests.post(f"{BASE_URL}/auth/login", data={"username": "testuser", "password": "password123"})
    if res.status_code == 200:
        return res.json()["access_token"]
    raise Exception(f"Failed to get token: {res.text}")

token = get_token()
headers = {"Authorization": f"Bearer {token}"}

paper_text = """
Deepchecks: Evaluating Retrieval-Augmented Generation (RAG) Systems.
ABSTRACT
Evaluating Retrieval-Augmented Generation (RAG) applications presents critical technical challenges due to hallucinations, inaccurate context retrieval, and complex pipelines. We introduce Deepchecks for RAG, a comprehensive evaluation framework designed to quantify retrieval precision, context relevance, answer faithfulness, and hallucination rates across RAG architectures.

1. INTRODUCTION & MAIN OBJECTIVE
The main objective of this paper is to introduce Deepchecks, an automated testing and continuous monitoring suite for RAG pipelines. Our primary goal is to provide researchers and engineers with actionable metrics to diagnose failure modes in context retrieval and LLM response generation.

2. PROPOSED METHODOLOGY & SYSTEM ARCHITECTURE
Deepchecks evaluates RAG systems through four core properties:
- Retrieval Precision: Measures whether the top-k retrieved passages contain true evidence.
- Context Relevance: Quantifies the ratio of relevant sentences within retrieved chunks.
- Answer Faithfulness: Verifies that every claim in the generated answer is directly supported by the retrieved context.
- Hallucination Rate: Computes the proportion of generated tokens containing unsupported facts.

3. DATASETS AND EXPERIMENTAL BENCHMARKS
We evaluate Deepchecks across three benchmark datasets: RAG-Bench-100 (100 synthetic QA pairs), WikiQA-Eval (500 multi-hop queries), and Medical-RAG-Corpus (250 domain-specific queries).

4. QUANTITATIVE RESULTS & BASELINE COMPARISON
In empirical trials, Deepchecks identified 94.2% of unfaithful claims and reduced false-positive hallucination detections by 18.5% compared to standard G-Eval baselines. On RAG-Bench-100, answer faithfulness evaluation achieved an F1 score of 0.91.

5. AUTHOR-STATED LIMITATIONS
The authors identify two key limitations: (1) high evaluation latency during multi-metric LLM-as-a-judge scoring, and (2) reliance on high-quality reference embeddings for domain-specific medical taxonomies.

6. FUTURE DIRECTIONS
Future work will focus on optimizing evaluation latency via lightweight local judge models and extending multi-modal RAG evaluation to visual and tabular contexts.
"""

test_questions = [
    "What is the main objective of this paper?",
    "What is the main research problem addressed by this paper?",
    "What is the proposed methodology?",
    "What dataset or benchmark was used?",
    "What were the quantitative results?",
    "What are the author-stated limitations?",
    "What future work do the authors suggest?",
    "What are the scientific contributions of this paper?",
]

for q in test_questions:
    print(f"\n==========================================")
    print(f"Testing Question: '{q}'")
    print(f"==========================================")
    payload = {
        "question": q,
        "uploaded_paper_name": "Deepchecks_RAG_Paper.pdf",
        "uploaded_paper_text": paper_text,
        "domain": "artificial_intelligence"
    }
    try:
        res = requests.post(f"{BASE_URL}/research/query", json=payload, headers=headers, timeout=25)
        if res.status_code == 200:
            data = res.json()
            answer = data.get("answer", "")
            citations = data.get("citations", {})
            print("Status: 200 OK")
            print("Answer Output:\n", answer)
            print("\nCitations returned:", list(citations.keys()))
        else:
            print(f"Error {res.status_code}: {res.text[:300]}")
    except Exception as e:
        print("Exception:", e)
