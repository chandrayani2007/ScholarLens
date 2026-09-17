import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath("."))

from src.pipeline.llm import get_llm_provider
from src.pipeline.rag import RAGPipeline, AntiCopyValidator
from scratch.verify_live_pipeline import OPEN_SCHOLAR_PAPER_TEXT

def main():
    llm = get_llm_provider("gemini")
    print(f"INITIALIZED LLM PROVIDER: {type(llm).__name__} (Model: {llm.model})")
    assert type(llm).__name__ == "GeminiLLMProvider", f"Expected GeminiLLMProvider, got {type(llm).__name__}"

    pipeline = RAGPipeline(llm_provider=llm)

    questions = [
        "What is the main research problem addressed by this paper?",
        "What methodology did the authors propose?",
        "What dataset or benchmark was used?",
        "How was the data prepared or preprocessed?",
        "What were the main quantitative results?",
        "How did the proposed approach compare with the baseline?",
        "What are the main contributions of this paper?",
        "Why did the authors choose this approach?",
        "What limitations did the authors identify?",
        "What is the main research problem, methodology, dataset, and main results?",
    ]

    results = []
    print("=" * 80)
    print("RUNNING LIVE GEMINI VERIFICATION ACROSS ALL 10 QUESTIONS ON SDC_BasePaper.pdf")
    print("=" * 80)

    for idx, q in enumerate(questions, 1):
        print(f"\n" + "-"*80)
        print(f"[{idx}/10] TESTING QUESTION: {q}")
        print("-"*80)

        res = pipeline.answer(
            question=q,
            uploaded_paper_text=OPEN_SCHOLAR_PAPER_TEXT,
            uploaded_paper_name="SDC_BasePaper.pdf",
            filters={"domain": "artificial_intelligence"},
        )

        is_synth, reason, overlap = AntiCopyValidator.validate_answer(res.answer, res.evidence)

        citations_list = list(res.citations.keys())
        has_citations = len(citations_list) > 0
        all_uploaded = all(cit.source_type == "uploaded" for cit in res.citations.values())

        print(f"ANTI-COPY VALIDATION: {'PASS' if is_synth else 'FAIL'} (score={overlap:.2f}, reason: {reason})")
        print(f"CITATIONS ({len(citations_list)}): {citations_list}")
        print(f"ALL CITATIONS FROM UPLOADED PAPER: {all_uploaded}")
        print(f"FINAL PROCESSED ANSWER:\n{res.answer}\n")

        results.append({
            "idx": idx,
            "question": q,
            "answer": res.answer,
            "evidence": res.evidence,
            "citations": res.citations,
            "is_synth": is_synth,
            "reason": reason,
            "overlap": overlap,
            "all_uploaded": all_uploaded,
            "has_citations": has_citations,
        })

    all_passed_anticopy = all(r["is_synth"] for r in results)
    all_passed_citations = all(r["has_citations"] and r["all_uploaded"] for r in results)

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS TABLE:")
    print("=" * 80)
    for r in results:
        status = "PASS" if (r["is_synth"] and r["has_citations"] and r["all_uploaded"]) else "FAIL"
        print(f"Q{r['idx']}: [{status}] Synth: {r['is_synth']} (overlap={r['overlap']:.2f}) | Cits: {list(r['citations'].keys())} | Q: {r['question'][:45]}...")

    print("=" * 80)
    print(f"OVERALL STATUS: {'ALL 10/10 PASSED' if (all_passed_anticopy and all_passed_citations) else 'SOME FAILED'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
