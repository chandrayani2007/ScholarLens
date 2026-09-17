import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

import logging
import json
from src.pipeline.llm import get_llm_provider
from src.pipeline.rag import RAGPipeline, QuestionRouter

logging.basicConfig(level=logging.INFO)

llm = get_llm_provider("gemini")
pipeline = RAGPipeline(llm_provider=llm)

ai038_questions = [
    {"num": 1, "paper_id": "AI038", "question": "What research problem does this paper address?"},
    {"num": 2, "paper_id": "AI038", "question": "What methodology does this paper propose?"},
    {"num": 3, "paper_id": "AI038", "question": "What are the main contributions of this paper?"},
    {"num": 4, "paper_id": "AI038", "question": "What are the main results?"},
    {"num": 5, "paper_id": "AI038", "question": "What limitations did the authors identify?"},
    {"num": 6, "paper_id": "AI038", "question": "Why did the authors choose this approach?"},
    {"num": 7, "paper_id": "AI038", "question": "What dataset or data did the paper use?"},
    {"num": 8, "paper_id": "AI038", "question": "Give a comprehensive summary of this paper."},
]

ai001_questions = [
    {"num": 9, "paper_id": "AI001", "question": "What research problem does this paper address?"},
    {"num": 10, "paper_id": "AI001", "question": "What methodology does this paper propose?"},
    {"num": 11, "paper_id": "AI001", "question": "What are the main contributions of this paper?"},
    {"num": 12, "paper_id": "AI001", "question": "What are the main results?"},
    {"num": 13, "paper_id": "AI001", "question": "Give a comprehensive summary of this paper."},
]

all_test_questions = ai038_questions + ai001_questions

reports = []
import time

for q_info in all_test_questions:
    num = q_info["num"]
    pid = q_info["paper_id"]
    qtext = q_info["question"]
    
    print("\n" + "="*80)
    print(f"TEST RUN {num}: Paper={pid} | Question='{qtext}'")
    print("="*80)
    
    time.sleep(2.5)
    filters = {"paper_id": pid}

    
    # 1. Routing
    target_mode, route_cat = QuestionRouter.classify(question=qtext, paper_id_filter=pid)
    
    # 2. Pipeline Execution
    res = pipeline.answer(
        question=qtext,
        filters=filters,
        top_k=10,
        request_id=f"VERIF-{num}"
    )
    
    answer_text = res.answer.strip()
    citations = list(res.citations.keys())
    ev_strength = res.why_this_answer.evidence_strength
    unsupported = res.why_this_answer.unsupported_claims
    
    # Verify zero cross-paper contamination
    cross_paper_contam = 0
    for cit_key, cit_obj in res.citations.items():
        if cit_obj.paper_id != pid and not cit_key.startswith("U"):
            cross_paper_contam += 1
            
    is_synthesis_success = bool(len(answer_text) > 40 and "insufficient evidence" not in answer_text.lower())
    
    report = {
        "num": num,
        "paper_id": pid,
        "question": qtext,
        "route": target_mode,
        "retrieved_count": res.retrieval_metadata.get("retrieved_count", 0),
        "selected_evidence_count": len(res.evidence),
        "synthesis": "SUCCESS" if is_synthesis_success else "FAIL",
        "answer_preview": answer_text[:150].replace("\n", " "),
        "full_answer": answer_text,
        "citations": ", ".join(citations) if citations else "None",
        "evidence_strength": ev_strength,
        "unsupported_claims": unsupported,
        "cross_paper_contamination": cross_paper_contam,
        "frontend": "DISPLAYED" if is_synthesis_success else "FAILED"
    }
    reports.append(report)
    
    safe_answer = answer_text.encode('ascii', 'ignore').decode('ascii')
    print(f"Paper: {pid}")
    print(f"Question: {qtext}")
    print(f"Route: {target_mode}")
    print(f"Initial retrieved chunks: {res.retrieval_metadata.get('retrieved_count', 0)}")
    print(f"Final selected evidence: {len(res.evidence)}")
    print(f"Gemini synthesis: {'SUCCESS' if is_synthesis_success else 'FAIL'}")
    print(f"Final answer:\n{safe_answer}")

    print(f"Citations: {', '.join(citations) if citations else 'None'}")
    print(f"Evidence strength: {ev_strength}")
    print(f"Unsupported claims: {unsupported}")
    print(f"Cross-paper contamination: {cross_paper_contam}")
    print(f"Frontend: {'DISPLAYED' if is_synthesis_success else 'FAILED'}")

print("\n" + "="*80)
print("VERIFICATION SUMMARY REPORT TABLE")
print("="*80)
for rep in reports:
    print(f"Paper: {rep['paper_id']} | Q: {rep['question']}")
    print(f"Route: {rep['route']} | Citations: {rep['citations']} | Strength: {rep['evidence_strength']} | Synthesis: {rep['synthesis']} | Cross-Paper Contam: {rep['cross_paper_contamination']}")
    print("-" * 80)
