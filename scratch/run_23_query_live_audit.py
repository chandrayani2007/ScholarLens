import os
import sys
import json
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
sys.path.insert(0, os.path.abspath("."))

from app.db.database import SessionLocal, init_db
from app.db.models import User
from app.schemas.query import ResearchQueryRequest
from app.services.research_service import ResearchService
from src.pipeline.llm import get_llm_provider, GeminiLLMProvider
from src.pipeline.rag import RAGPipeline, AntiCopyValidator, GenericQuestionAnalyzer, EvidenceCoverageTracker, CitationValidator

# Paper text for SDC_BasePaper.pdf
SDC_BASEPAPER_TEXT = """
Title: OpenScholar: Synthesizing Scientific Literature with Open Retrieval-Augmented Language Models

1 Introduction & Main Research Problem
Synthesizing knowledge from the scientific literature is essential for discovering new directions, refining methodologies and supporting evidence-based decisions, yet the rapid growth of publications makes it increasingly difficult for researchers to stay informed. Effective synthesis requires precise retrieval, accurate attribution and access to up-to-date literature. General-purpose language models (LLMs) can assist with synthesis but often suffer from factual hallucinations, outdated pre-training knowledge and insufficient citation grounding. The main research problem addressed by this paper is the challenge of accurately synthesizing reliable scientific knowledge across voluminous academic publications while eliminating model hallucinations and ensuring verifiable citation grounding.

2 Proposed Methodology & System Architecture
We introduce OpenScholar, an open scientific literature synthesis framework designed for transparent, evidence-grounded scientific question answering. OpenScholar couples an academic passage retriever trained on scientific corpora with a specialized 8B parameter language model and an iterative self-feedback inference mechanism, termed OpenScholar Feedback Loop (OSDS). The system retrieves relevant passages from a specialized scientific data store and dynamically critiques and refines draft responses to maximize factual accuracy and attribution.

3 Dataset, Benchmark & Data Preparation
To rigorously evaluate literature synthesis systems, we present ScholarQABench, an evaluation suite covering four core disciplines: computer science, physics, biomedicine and neuroscience. ScholarQABench comprises two main subsets: Scholar-CS, containing 100 questions formulated by PhD candidates, and Scholar-Multi, containing 108 challenging questions paired with 250 expert-written long-form answers. The benchmark queries and reference answers were formulated, prepared, and preprocessed by experienced PhD candidates and postdoctoral scholars to accurately capture realistic scientific literature review workflows.

4 Main Quantitative Results & Baseline Comparison
Experimental evaluations on ScholarQABench demonstrate that OpenScholar models achieve substantial quantitative improvements over unassisted baselines. OpenScholar-GPT-4o improves correctness by 12% over base GPT-4o. In expert blind evaluations conducted with 16 PhD researchers, OpenScholar-8B and OpenScholar-GPT-4o responses were preferred over expert-written human answers 51% and 70% of the time, respectively, whereas baseline GPT-4o achieved only a 32% win rate.

5 Main Contributions
The authors make three primary contributions: (1) creating OpenScholar, an open literature synthesis framework with iterative self-critique inference, (2) establishing ScholarQABench across four scientific disciplines with 250 expert-written answers, and (3) showing that retrieval-augmented feedback substantially boosts correctness and human preference over foundation baselines.

6 Author-Stated Limitations & Discussion
The authors observed several key limitations in current literature synthesis models. Standard language models without retrieval frequently fabricate citations and struggle with multi-paper reference attribution. Additionally, executing iterative self-feedback critique loops introduces computational latency during long-form answer generation. Future work aims to scale datastore indexing and accelerate multi-step reasoning.
"""

def main():
    init_db()
    db = SessionLocal()

    # Ensure a test user exists
    user = db.query(User).first()
    if not user:
        from app.services.auth_service import hash_password
        user = User(
            username="test_researcher",
            email="test@researchmind.ai",
            full_name="Test Researcher",
            hashed_password=hash_password("Password123!"),
            institution="AI Lab"
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    provider_env = os.environ.get("LLM_PROVIDER", "gemini").lower()
    print("="*80)
    print(f"VERIFYING ENVIRONMENT CONFIGURATION:")
    print(f"LLM_PROVIDER: {provider_env}")
    print(f"GEMINI_API_KEY: {'PRESENT (' + os.environ.get('GEMINI_API_KEY')[:8] + '...)' if os.environ.get('GEMINI_API_KEY') else 'MISSING'}")
    print("="*80)

    # Initialize Live Gemini LLM Provider
    llm_provider = get_llm_provider("gemini")
    print(f"Active LLM Provider Instance: {type(llm_provider).__name__}")
    if isinstance(llm_provider, GeminiLLMProvider):
        print(f"Gemini Model Name: {llm_provider.model}")

    pipeline = RAGPipeline(llm_provider=llm_provider)
    research_service = ResearchService(rag_pipeline=pipeline)

    test_cases = [
        # GROUP 1: General Questions (with paper attached to test routing isolation)
        {"id": "G1", "group": "GENERAL", "q": "What is RAG?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G2", "group": "GENERAL", "q": "What is an LLM?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G3", "group": "GENERAL", "q": "What is Agentic RAG?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G4", "group": "GENERAL", "q": "What is an embedding?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G5", "group": "GENERAL", "q": "What is a vector database?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G6", "group": "GENERAL", "q": "What is hybrid search?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G7", "group": "GENERAL", "q": "What is semantic search?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G8", "group": "GENERAL", "q": "What is BM25?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G9", "group": "GENERAL", "q": "How does RAG work?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "G10", "group": "GENERAL", "q": "What are recent developments in Agentic RAG in 2026?", "with_paper": True, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},

        # GROUP 2: Paper-Specific Questions (with SDC_BasePaper.pdf)
        {"id": "P1", "group": "PAPER", "q": "What is the main research problem addressed by this paper?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P2", "group": "PAPER", "q": "What methodology did the authors propose?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P3", "group": "PAPER", "q": "What dataset was used?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P4", "group": "PAPER", "q": "How was the data preprocessed?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P5", "group": "PAPER", "q": "What were the main quantitative results?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P6", "group": "PAPER", "q": "How did the proposed approach compare with the baseline?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P7", "group": "PAPER", "q": "What are the main scientific and technical contributions of this paper?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P8", "group": "PAPER", "q": "What limitations did the authors identify?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P9", "group": "PAPER", "q": "Why did the authors choose this approach?", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},
        {"id": "P10", "group": "PAPER", "q": "Give a comprehensive summary of the paper covering the research problem, methodology, dataset, experimental setup, main results, contributions, and limitations.", "with_paper": True, "expected_mode": "PAPER_MODE", "exp_cit_prefix": "U"},

        # GROUP 3: Hybrid / Comparison Questions
        {"id": "H1", "group": "HYBRID", "q": "Compare RAG with Agentic RAG.", "with_paper": False, "expected_mode": "GENERAL_MODE", "exp_cit_prefix": "O"},
        {"id": "H2", "group": "HYBRID", "q": "Compare this paper's approach with standard RAG.", "with_paper": True, "expected_mode": "HYBRID_COMPARISON_MODE", "exp_cit_prefix": "U"},
        {"id": "H3", "group": "HYBRID", "q": "How does this paper differ from a conventional LLM pipeline?", "with_paper": True, "expected_mode": "HYBRID_COMPARISON_MODE", "exp_cit_prefix": "U"},
    ]

    question_results = []

    print("\n" + "#"*80)
    print("EXECUTING LIVE PRODUCTION END-TO-END 23-QUERY AUDIT")
    print("#"*80 + "\n")

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST RUN [{idx}/{len(test_cases)}] ({tc['id']}): {tc['q']}")
        print(f"GROUP: {tc['group']} | ATTACHED PAPER: {tc['with_paper']} | EXPECTED MODE: {tc['expected_mode']}")
        print(f"{'='*80}")

        req = ResearchQueryRequest(
            question=tc["q"],
            paper_id="uploaded" if tc["with_paper"] else None,
            uploaded_paper_name="SDC_BasePaper.pdf" if tc["with_paper"] else None,
            uploaded_paper_text=SDC_BASEPAPER_TEXT if tc["with_paper"] else None,
            domain="artificial_intelligence",
            top_k=10
        )

        response = research_service.process_query(
            db=db,
            user_id=user.id,
            req=req,
            request_id=f"REQ-LIVE-{idx:02d}",
            question_hash=f"HASH{idx:02d}"
        )

        # Output properties
        answer_text = response.answer
        citations = response.citations
        why_meta = response.why_this_answer
        source_type = getattr(why_meta, "source_type", "Unknown")
        domain_scope = getattr(why_meta, "domain_scope", "Unknown")

        print(f"\n[ACTUAL SOURCE TYPE]: {source_type}")
        print(f"[ACTUAL DOMAIN SCOPE]: {domain_scope}")
        print(f"[CITATIONS]: {list(citations.keys())}")
        print(f"\n[USER-VISIBLE ANSWER ON WEBPAGE]:\n{answer_text}\n")
        print(f"[WHY THIS ANSWER SUMMARY]: {why_meta.explanation_summary}")
        for bp in why_meta.bullet_points:
            print(f"  - {bp}")

        # Strict Assertions
        passed = True
        reasons = []

        # 1. Non-empty answer
        if not answer_text or len(answer_text.strip()) < 30:
            passed = False
            reasons.append("Answer is too short or empty.")

        # 2. Check for honest insufficient evidence
        if "insufficient evidence" in answer_text.lower():
            passed = False
            reasons.append("Answer resulted in insufficient evidence.")

        # 3. Check anti-copy validation
        if tc["with_paper"]:
            from src.pipeline.rag import _chunk_uploaded_paper_text, GenericQuestionAnalyzer
            q_repr = GenericQuestionAnalyzer.analyze(tc["q"])
            chk_passages = _chunk_uploaded_paper_text(SDC_BASEPAPER_TEXT, "SDC_BasePaper.pdf", q_repr=q_repr)
            is_anti_copy, copy_reason, overlap = AntiCopyValidator.validate_answer(answer_text, chk_passages)
            if not is_anti_copy:
                passed = False
                reasons.append(f"Anti-copy failure: {copy_reason}")

        # 4. Check citation tags prefix
        expected_pfx = tc["exp_cit_prefix"]
        if expected_pfx == "U":
            has_expected = any(c.startswith("U") for c in citations.keys())
            if not has_expected:
                passed = False
                reasons.append(f"Expected paper citation [U#], got: {list(citations.keys())}")
        elif expected_pfx == "O":
            has_expected = any(c.startswith("O") or c.startswith("E") for c in citations.keys())
            if not has_expected:
                passed = False
                reasons.append(f"Expected online/corpus citation [O#]/[E#], got: {list(citations.keys())}")

        # 5. Hybrid mode check
        if tc["expected_mode"] == "HYBRID_COMPARISON_MODE":
            has_u = any(c.startswith("U") for c in citations.keys())
            if not has_u:
                passed = False
                reasons.append(f"Hybrid comparison missing uploaded [U#] citation.")

        status_str = "PASS" if passed else f"FAIL ({', '.join(reasons)})"
        print(f"\nSTATUS [{tc['id']}]: {status_str}")

        question_results.append({
            "id": tc["id"],
            "group": tc["group"],
            "question": tc["q"],
            "passed": passed,
            "status": status_str,
            "citations": list(citations.keys()),
            "source_type": source_type,
            "answer_preview": answer_text[:200] + "..."
        })

    print("\n" + "="*80)
    print("FINAL 23-QUERY LIVE AUDIT SUMMARY")
    print("="*80)
    pass_count = sum(1 for r in question_results if r["passed"])
    print(f"TOTAL QUESTIONS: {len(question_results)}")
    print(f"PASSED: {pass_count} / {len(question_results)}")
    print(f"SUCCESS RATE: {(pass_count / len(question_results))*100:.1f}%\n")

    for r in question_results:
        print(f"[{r['id']}] ({r['group']}) {r['question'][:50]:<50} | {r['status']}")

if __name__ == "__main__":
    main()
