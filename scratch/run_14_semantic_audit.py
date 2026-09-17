"""
Live 14-Query Semantic Audit — Final Semantic Quality Fix Verification
Uses the same ResearchService.process_query(db, user_id, req) pattern as the
working 23-query audit.
"""
import os, sys, re, traceback
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.abspath("."))

from app.db.database import SessionLocal, init_db
from app.db.models import User
from app.schemas.query import ResearchQueryRequest
from app.services.research_service import ResearchService
from src.pipeline.llm import get_llm_provider, GeminiLLMProvider
from src.pipeline.rag import RAGPipeline

# ── Test cases ────────────────────────────────────────────────────────────────
TESTS = [
    ("G1",  "GENERAL", "What is RAG?",                                          True,  "GENERAL_MODE"),
    ("G2",  "GENERAL", "How does RAG work?",                                    True,  "GENERAL_MODE"),
    ("G3",  "GENERAL", "What is Agentic RAG?",                                  True,  "GENERAL_MODE"),
    ("G4",  "GENERAL", "What is the difference between RAG and Agentic RAG?",   True,  "GENERAL_MODE"),
    ("G5",  "GENERAL", "What is BM25?",                                         True,  "GENERAL_MODE"),
    ("G6",  "GENERAL", "What is semantic search?",                              True,  "GENERAL_MODE"),
    ("G7",  "GENERAL", "What is a vector database?",                            True,  "GENERAL_MODE"),
    ("P1",  "PAPER",   "What is the main research problem addressed by this paper?", True, "PAPER_MODE"),
    ("P2",  "PAPER",   "What methodology did the authors propose?",             True,  "PAPER_MODE"),
    ("P3",  "PAPER",   "What dataset was used?",                                True,  "PAPER_MODE"),
    ("P4",  "PAPER",   "What were the main quantitative results?",              True,  "PAPER_MODE"),
    ("P5",  "PAPER",   "What limitations did the authors identify?",            True,  "PAPER_MODE"),
    ("H1",  "HYBRID",  "Compare this paper's approach with standard RAG.",      True,  "HYBRID_COMPARISON_MODE"),
    ("H2",  "HYBRID",  "Compare RAG with Agentic RAG.",                         False, "GENERAL_MODE"),
]

# Use the same embedded paper text that the working 23-query audit uses
SDC_BASEPAPER_TEXT = """
Title: OpenScholar: Synthesizing Scientific Literature with Open Retrieval-Augmented Language Models

1 Introduction & Main Research Problem
Synthesizing knowledge from the scientific literature is essential for discovering new directions, refining methodologies and supporting evidence-based decisions, yet the rapid growth of publications makes it increasingly difficult for researchers to stay informed. Effective synthesis requires precise retrieval, accurate attribution and access to up-to-date literature. General-purpose language models (LLMs) can assist with synthesis but often suffer from factual hallucinations, outdated pre-training knowledge and insufficient citation grounding. The main research problem addressed by this paper is the challenge of accurately synthesizing reliable scientific knowledge across voluminous academic publications while eliminating model hallucinations and ensuring verifiable citation grounding.

2 Proposed Methodology & System Architecture
We introduce OpenScholar, an open scientific literature synthesis framework designed for transparent, evidence-grounded scientific question answering. OpenScholar couples an academic passage retriever trained on scientific corpora with a specialized 8B parameter language model and an iterative self-feedback inference mechanism, termed OpenScholar Feedback Loop (OSDS). The system retrieves relevant passages from a specialized scientific data store and dynamically critiques and refines draft responses to maximize factual accuracy and attribution.

3 Dataset, Benchmark & Data Preparation
To rigorously evaluate literature synthesis systems, we present ScholarQABench, an evaluation suite covering four core disciplines: computer science, physics, biomedicine and neuroscience. ScholarQABench comprises two main subsets: Scholar-CS, containing 100 questions formulated by PhD candidates, and Scholar-Multi, containing 108 challenging questions paired with 250 expert-written long-form answers.

4 Main Quantitative Results & Baseline Comparison
OpenScholar-GPT-4o improves correctness by 12% over base GPT-4o. In expert blind evaluations conducted with 16 PhD researchers, OpenScholar-8B and OpenScholar-GPT-4o responses were preferred over expert-written human answers 51% and 70% of the time, respectively, whereas baseline GPT-4o achieved only a 32% win rate.

5 Main Contributions
The authors make three primary contributions: (1) creating OpenScholar, an open literature synthesis framework with iterative self-critique inference, (2) establishing ScholarQABench across four scientific disciplines with 250 expert-written answers, and (3) showing that retrieval-augmented feedback substantially boosts correctness and human preference over foundation baselines.

6 Author-Stated Limitations & Discussion
The authors observed several key limitations. Standard language models without retrieval frequently fabricate citations and struggle with multi-paper reference attribution. Additionally, executing iterative self-feedback critique loops introduces computational latency during long-form answer generation. Future work aims to scale datastore indexing and accelerate multi-step reasoning.
"""

# ── Semantic checks ───────────────────────────────────────────────────────────
BANNED_OPENERS = [
    "the paper addresses several",
    "the study investigates",
    "this paper explores",
    "experimental evaluations demonstrate reported",
    "the authors identify various",
    "this study explores",
    "measurable improvements",
    "measurable performance improvements",
]

GENERIC_QUANTITATIVE_PHRASES = [
    "measurable improvements",
    "measurable performance improvements",
    "outperforms baselines",
    "demonstrates improvements",
]

def check_numbered_list_corruption(answer: str):
    m = re.search(r'\d+\.\s+\d+\.', answer)
    if m:
        return False, f"Corruption: '...{answer[max(0,m.start()-5):m.end()+40]}...'"
    return True, "OK"

def check_answer_first(answer: str):
    lines = [l.strip() for l in answer.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return False, "No content lines"
    first = lines[0].lower()
    for banned in BANNED_OPENERS:
        if first.startswith(banned):
            return False, f"Banned opener: '{lines[0][:120]}'"
    return True, "OK"

def check_no_generic_quantitative(answer: str):
    al = answer.lower()
    for phrase in GENERIC_QUANTITATIVE_PHRASES:
        if phrase in al:
            return False, f"Generic phrase: '{phrase}'"
    return True, "OK"

def check_evidence_strength_vs_unsupported(strength: str, unsupported: int):
    if unsupported > 0 and strength == "High":
        return False, f"strength={strength} but unsupported={unsupported} (must be <= Moderate)"
    return True, "OK"

# ── Setup ─────────────────────────────────────────────────────────────────────
init_db()
db = SessionLocal()

user = db.query(User).first()
if not user:
    from app.services.auth_service import hash_password
    user = User(
        username="test_researcher", email="test@researchmind.ai",
        full_name="Test Researcher", hashed_password=hash_password("Password123!"),
        institution="AI Lab"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

llm_provider = get_llm_provider("gemini")
pipeline = RAGPipeline(llm_provider=llm_provider)
research_service = ResearchService(rag_pipeline=pipeline)

passed = 0
failed = 0
results = []

print("=" * 80)
print("LIVE 14-QUERY SEMANTIC QUALITY AUDIT")
print("=" * 80)

for idx, (tid, group, question, use_paper, expected_mode) in enumerate(TESTS, 1):
    print(f"\n{'='*80}")
    print(f"TEST [{idx}/{len(TESTS)}] ({tid}) ({group}): {question}")
    print(f"PAPER ATTACHED: {use_paper} | EXPECTED MODE: {expected_mode}")
    print("=" * 80)

    try:
        req = ResearchQueryRequest(
            question=question,
            paper_id="uploaded" if use_paper else None,
            uploaded_paper_name="SDC_BasePaper.pdf" if use_paper else None,
            uploaded_paper_text=SDC_BASEPAPER_TEXT if use_paper else None,
            domain="artificial_intelligence",
            top_k=10
        )

        response = research_service.process_query(
            db=db,
            user_id=user.id,
            req=req,
            request_id=f"AUDIT-{tid}",
            question_hash=f"HASH{idx:02d}"
        )

        answer = response.answer
        citations = list(response.citations.keys()) if response.citations else []
        why = response.why_this_answer
        unsupported = getattr(why, "unsupported_claims", 0)
        evidence_strength = getattr(why, "evidence_strength", "?")
        source_type = getattr(why, "source_type", "?")

        # Detect mode from source label
        if source_type and "Uploaded Paper" in source_type and "Online" in source_type:
            actual_mode = "HYBRID_COMPARISON_MODE"
        elif source_type and "Uploaded Paper" in source_type:
            actual_mode = "PAPER_MODE"
        else:
            actual_mode = "GENERAL_MODE"

        # Run checks
        list_ok, list_msg = check_numbered_list_corruption(answer)
        first_ok, first_msg = check_answer_first(answer)
        quant_ok, quant_msg = check_no_generic_quantitative(answer) if group == "PAPER" else (True, "N/A")
        strength_ok, strength_msg = check_evidence_strength_vs_unsupported(evidence_strength, unsupported)
        mode_ok = (actual_mode == expected_mode)

        all_checks = [
            ("Mode routing",                mode_ok,     f"Expected {expected_mode}, got {actual_mode}"),
            ("Numbered list format",        list_ok,     list_msg),
            ("Answer-first sentence",       first_ok,    first_msg),
            ("No generic quantitative",     quant_ok,    quant_msg),
            ("Evidence strength consistent",strength_ok, strength_msg),
        ]

        test_pass = all(ok for _, ok, _ in all_checks)

        print(f"\n[ACTUAL MODE]:     {actual_mode}")
        print(f"[SOURCE TYPE]:     {source_type}")
        print(f"[CITATIONS]:       {citations}")
        print(f"[UNSUPPORTED]:     {unsupported}")
        print(f"[EVIDENCE STR]:    {evidence_strength}")
        print(f"\n[ANSWER]:\n{answer[:900]}{'...' if len(answer) > 900 else ''}")
        print(f"\n[CHECKS]:")
        for name, ok, msg in all_checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {'' if ok else msg}")

        status = "PASS" if test_pass else "FAIL"
        print(f"\nSTATUS [{tid}]: {status}")
        if test_pass:
            passed += 1
        else:
            failed += 1
        results.append((tid, group, question, actual_mode, evidence_strength, unsupported, status))

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        failed += 1
        results.append((tid, group, question, "ERROR", "?", 0, "FAIL"))

print("\n" + "=" * 80)
print("FINAL 14-QUERY SEMANTIC AUDIT SUMMARY")
print("=" * 80)
print(f"TOTAL: {len(TESTS)} | PASSED: {passed} | FAILED: {failed}")
print(f"SUCCESS RATE: {100*passed/len(TESTS):.1f}%\n")
for tid, group, question, mode, strength, unsup, status in results:
    q_display = (question[:52] + "...") if len(question) > 55 else question.ljust(55)
    print(f"[{tid}] ({group:<6}) {q_display} | {mode:<25} | Str:{strength:<11} | Unsup:{unsup} | {status}")
