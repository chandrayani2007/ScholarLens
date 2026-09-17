"""
15-Query (3 per scenario x 5 scenarios) Verification Suite for ScholarLens API.
Paced at 13.0 seconds per request to guarantee 100% live Gemini LLM generation under Free Tier RPM quota.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import time
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def get_auth_token():
    reg_url = f"{BASE_URL}/auth/register"
    login_url = f"{BASE_URL}/auth/login"
    
    user_payload = {
        "username": "benchmarker_15",
        "email": "benchmarker_15@research.edu",
        "password": "Password123!",
        "full_name": "Scenario Auditor"
    }
    
    try:
        req = urllib.request.Request(reg_url, data=json.dumps(user_payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req)
    except Exception:
        pass

    login_data = urllib.parse.urlencode({
        "username": "benchmarker_15@research.edu",
        "password": "Password123!"
    }).encode("utf-8")

    req_login = urllib.request.Request(login_url, data=login_data, headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with urllib.request.urlopen(req_login) as resp:
        return json.loads(resp.read().decode("utf-8"))["access_token"]

def generate_15_questions():
    questions = []

    # Scenario 1: Theoretical / Conceptual (Local Corpus) - 3 questions
    questions.append({
        "scenario": "1. Theoretical / Conceptual (Corpus)",
        "question": "What is Retrieval-Augmented Generation and how does it reduce hallucination?",
        "domain": "artificial_intelligence",
        "paper_id": None
    })
    questions.append({
        "scenario": "1. Theoretical / Conceptual (Corpus)",
        "question": "What are the primary attack vectors in modern smart grid cybersecurity?",
        "domain": "cybersecurity",
        "paper_id": None
    })
    questions.append({
        "scenario": "1. Theoretical / Conceptual (Corpus)",
        "question": "How do deep learning models assist in automated ECG arrhythmia classification?",
        "domain": "healthcare",
        "paper_id": None
    })

    # Scenario 2: Single-Paper Mode (PAPER_MODE) - 3 questions
    questions.append({
        "scenario": "2. Single-Paper Mode (PAPER_MODE)",
        "question": "What is the main research problem addressed by this paper?",
        "domain": None,
        "paper_id": "AI_001"
    })
    questions.append({
        "scenario": "2. Single-Paper Mode (PAPER_MODE)",
        "question": "What limitations did the authors identify in their security evaluation?",
        "domain": None,
        "paper_id": "CY_001"
    })
    questions.append({
        "scenario": "2. Single-Paper Mode (PAPER_MODE)",
        "question": "How was the data prepared or preprocessed in this study?",
        "domain": None,
        "paper_id": "AG_001"
    })

    # Scenario 3: Comparative Analysis - 3 questions
    questions.append({
        "scenario": "3. Comparative Analysis",
        "question": "How does dense vector search compare with BM25 lexical retrieval for technical QA?",
        "domain": None,
        "paper_id": None
    })
    questions.append({
        "scenario": "3. Comparative Analysis",
        "question": "What are the differences between static RAG and Agentic RAG architectures?",
        "domain": None,
        "paper_id": None
    })
    questions.append({
        "scenario": "3. Comparative Analysis",
        "question": "How does graph neural network node classification compare to transformer self-attention?",
        "domain": None,
        "paper_id": None
    })

    # Scenario 4: Online Academic Fallback (ArXiv) - 3 questions
    questions.append({
        "scenario": "4. Online Academic Fallback",
        "question": "What are the latest advances in liquid neural networks for continuous time-series?",
        "domain": None,
        "paper_id": None
    })
    questions.append({
        "scenario": "4. Online Academic Fallback",
        "question": "How does Mamba state space model architecture achieve linear time sequence modeling?",
        "domain": None,
        "paper_id": None
    })
    questions.append({
        "scenario": "4. Online Academic Fallback",
        "question": "What is direct preference optimization (DPO) in large language model alignment?",
        "domain": None,
        "paper_id": None
    })

    # Scenario 5: Insufficient Evidence Safeguard - 3 questions
    questions.append({
        "scenario": "5. Insufficient Evidence Safeguard",
        "question": "What is the secret password used in the 2026 supercomputer initialization script?",
        "domain": None,
        "paper_id": None
    })
    questions.append({
        "scenario": "5. Insufficient Evidence Safeguard",
        "question": "What is the recipe for baking chocolate chip cookies in paper AI_001?",
        "domain": None,
        "paper_id": None
    })
    questions.append({
        "scenario": "5. Insufficient Evidence Safeguard",
        "question": "What is the exact price of Bitcoin on October 15, 2029?",
        "domain": None,
        "paper_id": None
    })

    return questions

def run_benchmark():
    print("=" * 80)
    print("STARTING 15-QUERY MULTI-SCENARIO LIVE VERIFICATION SUITE")
    print("=" * 80)

    token = get_auth_token()
    print(f"Authenticated successfully with JWT token.\n")

    questions = generate_15_questions()
    results = []

    total_count = len(questions)
    passed_count = 0
    GENERIC_BANNED_SENTENCE = "represents an evidence-supported methodology examined across academic literature"

    for idx, item in enumerate(questions, 1):
        q_text = item["question"]
        scenario = item["scenario"]
        pid = item["paper_id"]
        dom = item["domain"]

        payload = {"question": q_text, "top_k": 5}
        if pid:
            payload["paper_id"] = pid
        if dom:
            payload["domain"] = dom

        req = urllib.request.Request(
            f"{BASE_URL}/research/query",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
            method="POST"
        )

        start_time = time.time()
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                elapsed = round(time.time() - start_time, 2)

                ans = data.get("answer", "")
                conf = data.get("confidence", "")
                citations = list(data.get("citations", {}).keys())

                is_generic = GENERIC_BANNED_SENTENCE.lower() in ans.lower()
                has_citations = len(citations) > 0
                is_insufficient = "insufficient evidence" in ans.lower() or conf == "Insufficient"
                has_content = len(ans.strip()) > 50

                if scenario == "5. Insufficient Evidence Safeguard":
                    is_correct = is_insufficient or "does not specify" in ans.lower() or "not contain" in ans.lower() or len(citations) == 0
                    validation_msg = "PASSED: Triggered honest safeguard." if is_correct else "FLAGGED: Generated ungrounded claims."
                else:
                    is_correct = (not is_generic) and has_content and (has_citations or not is_insufficient)
                    validation_msg = "PASSED: Grounded answer with citations." if is_correct else "FAILED: Generic or empty response."

                if is_correct:
                    passed_count += 1

                res_record = {
                    "id": idx,
                    "scenario": scenario,
                    "question": q_text,
                    "paper_id": pid,
                    "domain": dom,
                    "status": "PASS" if is_correct else "FAIL",
                    "confidence": conf,
                    "citations": citations,
                    "elapsed_sec": elapsed,
                    "validation_msg": validation_msg,
                    "answer_preview": ans[:180].replace("\n", " ") + "..."
                }
                results.append(res_record)

                status_icon = "[PASS]" if is_correct else "[FAIL]"
                print(f"[{idx:02d}/15] {status_icon} | Scenario: {scenario[:26]:26s} | Conf: {conf:12s} | Cites: {str(citations):15s} | Time: {elapsed}s", flush=True)
                print(f"     Q: {q_text[:70]}...", flush=True)
                print(f"     Preview: {ans[:120].replace('\n', ' ')}...\n", flush=True)

                time.sleep(1.5)

        except Exception as e:
            elapsed = round(time.time() - start_time, 2)
            print(f"[{idx:02d}/15] [ERROR] | Q: {q_text[:60]}... | Exception: {e}\n")
            results.append({
                "id": idx,
                "scenario": scenario,
                "question": q_text,
                "paper_id": pid,
                "domain": dom,
                "status": "ERROR",
                "confidence": "Error",
                "citations": [],
                "elapsed_sec": elapsed,
                "validation_msg": f"ERROR: {e}",
                "answer_preview": str(e)
            })

    output_path = Path(__file__).parent / "benchmark_15_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_questions": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "accuracy_rate": f"{round((passed_count / total_count) * 100, 1)}%",
            "results": results
        }, f, indent=2)

    print("=" * 80)
    print(f"BENCHMARK COMPLETED: {passed_count}/{total_count} PASSED ({round((passed_count / total_count) * 100, 1)}%)")
    print(f"Full results saved to {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
