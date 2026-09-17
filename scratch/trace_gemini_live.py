import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath("."))

from src.pipeline.llm import get_llm_provider, GeminiLLMProvider
from src.pipeline.rag import RAGPipeline, AntiCopyValidator

def main():
    with open("scratch/verify_live_pipeline.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Get the OPEN_SCHOLAR_PAPER_TEXT
    from scratch.verify_live_pipeline import OPEN_SCHOLAR_PAPER_TEXT

    llm = get_llm_provider("gemini")
    print(f"INITIALIZED LLM PROVIDER: {type(llm).__name__}")
    print(f"API KEY PRESENT: {bool(llm.api_key)}")

    pipeline = RAGPipeline(llm_provider=llm)

    q = "What is the main research problem addressed by this paper?"

    print("\n" + "="*80)
    print("EXECUTING LIVE QUERY THROUGH RAGPipeline (with Gemini)...")
    print("="*80)

    res = pipeline.answer(
        question=q,
        uploaded_paper_text=OPEN_SCHOLAR_PAPER_TEXT,
        uploaded_paper_name="SDC_BasePaper.pdf",
        filters={"domain": "artificial_intelligence"},
    )

    print("\n" + "="*80)
    print("RESULT RECEIVED:")
    print("="*80)
    print(f"ANSWER:\n{res.answer}\n")
    print(f"CITATIONS: {res.citations}")
    is_synth, reason, overlap = AntiCopyValidator.validate_answer(res.answer, res.evidence)
    print(f"ANTI-COPY VALIDATION: {is_synth} (score={overlap:.2f}, reason: {reason})")

if __name__ == "__main__":
    main()
