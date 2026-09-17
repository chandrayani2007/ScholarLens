import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

import logging
import json
import time
from src.pipeline.llm import get_llm_provider
from src.pipeline.rag import RAGPipeline, QuestionRouter

logging.basicConfig(level=logging.INFO)

llm = get_llm_provider("gemini")
pipeline = RAGPipeline(llm_provider=llm)

papers = ["AI038", "AI001", "CY001", "HC001", "CL001", "AG001"]
questions_template = [
    {"q_idx": 1, "intent": "ResearchProblem", "text": "What research problem does this paper address?"},
    {"q_idx": 2, "intent": "Methodology", "text": "What methodology does this paper propose?"},
    {"q_idx": 3, "intent": "Dataset", "text": "What dataset or data did the paper use?"},
    {"q_idx": 4, "intent": "Finding", "text": "What are the main results?"},
    {"q_idx": 5, "intent": "Contribution", "text": "What are the main contributions of this paper?"},
    {"q_idx": 6, "intent": "Limitation", "text": "What limitations did the authors identify?"},
    {"q_idx": 7, "intent": "Comprehensive", "text": "Give a comprehensive summary of this paper."},
]

test_cases = []
case_counter = 1
for p in papers:
    for q in questions_template:
        test_cases.append({
            "case_id": case_counter,
            "paper_id": p,
            "q_idx": q["q_idx"],
            "intent": q["intent"],
            "question": q["text"]
        })
        case_counter += 1

print(f"Total test cases to execute: {len(test_cases)}")

reports = []
total_questions = len(test_cases)
successful_answers = 0
insufficient_answers = 0
total_unsupported_claims = 0
total_partially_supported_claims = 0
total_cross_paper_contam = 0
gemini_failures = 0
frontend_failures = 0
loading_hanging_requests = 0

start_time = time.time()

for tc in test_cases:
    cid = tc["case_id"]
    pid = tc["paper_id"]
    qtext = tc["question"]
    
    print("\n" + "="*80)
    print(f"AUDIT RUN {cid}/{total_questions}: Paper={pid} | Question='{qtext}'")
    print("="*80)
    
    time.sleep(2.0)  # Pacing for API rate limits
    filters = {"paper_id": pid}
    
    try:
        target_mode, route_cat = QuestionRouter.classify(question=qtext, paper_id_filter=pid)
        
        res = pipeline.answer(
            question=qtext,
            filters=filters,
            top_k=10,
            request_id=f"AUDIT-{cid}"
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
                
        is_insufficient = "insufficient evidence" in answer_text.lower() or ev_strength == "Insufficient"
        is_displayed = bool(len(answer_text) > 20)
        
        if is_insufficient:
            insufficient_answers += 1
        else:
            successful_answers += 1
            
        total_unsupported_claims += unsupported
        total_cross_paper_contam += cross_paper_contam
        
        if not is_displayed:
            frontend_failures += 1
            
        report = {
            "case_id": cid,
            "paper_id": pid,
            "question": qtext,
            "route": target_mode,
            "retrieved_count": res.retrieval_metadata.get("retrieved_count", 0),
            "selected_evidence_count": len(res.evidence),
            "synthesis": "SUCCESS" if is_displayed else "FAIL",
            "is_insufficient": is_insufficient,
            "answer_preview": answer_text[:120].replace("\n", " "),
            "full_answer": answer_text,
            "citations": ", ".join(citations) if citations else "None",
            "evidence_strength": ev_strength,
            "unsupported_claims": unsupported,
            "cross_paper_contamination": cross_paper_contam,
            "frontend": "DISPLAYED" if is_displayed else "FAILED"
        }
        reports.append(report)
        
        safe_answer = answer_text.encode('ascii', 'ignore').decode('ascii')
        print(f"Paper: {pid}")
        print(f"Question: {qtext}")
        print(f"Route: {target_mode}")
        print(f"Initial retrieved chunks: {res.retrieval_metadata.get('retrieved_count', 0)}")
        print(f"Final selected evidence: {len(res.evidence)}")
        print(f"Citations: {', '.join(citations) if citations else 'None'}")
        print(f"Evidence strength: {ev_strength}")
        print(f"Unsupported claims: {unsupported}")
        print(f"Cross-paper contamination: {cross_paper_contam}")
        print(f"Frontend: {'DISPLAYED' if is_displayed else 'FAILED'}")
        print(f"Answer:\n{safe_answer}")

    except Exception as e:
        print(f"ERROR executing test case {cid}: {e}")
        gemini_failures += 1
        frontend_failures += 1

print("\n" + "="*80)
print("FINAL AUDIT SUMMARY REPORT")
print("="*80)
print(f"Total questions: {total_questions}")
print(f"Successful answers: {successful_answers}")
print(f"Insufficient-evidence answers: {insufficient_answers}")
print(f"Unsupported claims in displayed answers: {total_unsupported_claims}")
print(f"Partially supported claims: {total_partially_supported_claims}")
print(f"Cross-paper contamination: {total_cross_paper_contam}")
print(f"Gemini failures: {gemini_failures}")
print(f"Frontend failures: {frontend_failures}")
print(f"Loading/hanging requests: {loading_hanging_requests}")
print(f"Total audit elapsed time: {time.time() - start_time:.2f} seconds")
