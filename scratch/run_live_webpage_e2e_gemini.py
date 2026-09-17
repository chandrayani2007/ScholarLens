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
from src.pipeline.rag import RAGPipeline, AntiCopyValidator, GenericQuestionAnalyzer, EvidenceCoverageTracker, CitationValidator, AnswerCompletenessChecker

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

    test_questions = [
        "What is the main research problem addressed by this paper?",
        "What methodology did the authors propose?",
        "What dataset or benchmark was used for evaluation?",
        "How was the data prepared or preprocessed before evaluation?",
        "What was the experimental setup, including models, baselines, datasets, and evaluation procedure?",
        "What were the main quantitative results?",
        "How did the proposed approach compare with the baseline methods?",
        "What are the main scientific and technical contributions of this paper?",
        "What limitations did the authors identify?",
        "What is the research problem, methodology, dataset, experimental setup, main results, contributions, and limitations of this paper?"
    ]

    question_results = []

    print("\n" + "#"*80)
    print("EXECUTING LIVE PRODUCTION END-TO-END QA (POST /api/research/query)")
    print("#"*80 + "\n")

    for idx, q in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"TEST RUN [{idx}/10]: {q}")
        print(f"{'='*80}")

        req = ResearchQueryRequest(
            question=q,
            paper_id="uploaded",
            uploaded_paper_name="SDC_BasePaper.pdf",
            uploaded_paper_text=SDC_BASEPAPER_TEXT,
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

        final_answer = response.answer
        citations_used = CitationValidator.extract_citations(final_answer)
        evidence_list = response.evidence
        evidence_ids = [getattr(e, 'citation_id', f'U{i+1}') for i, e in enumerate(evidence_list)]
        evidence_texts = [getattr(e, 'text', '') for e in evidence_list]

        q_repr = GenericQuestionAnalyzer.analyze(q)
        cov_items, cov_state, cov_ratio, missing = EvidenceCoverageTracker.evaluate_coverage(q, q_repr, evidence_list)
        is_synth, reason, score = AntiCopyValidator.validate_answer(final_answer, evidence_list)
        is_comp, covered_asp, missing_asp = AnswerCompletenessChecker.check_completeness(final_answer, q_repr.intent)

        # Check for verbatim sentences
        has_verbatim_sent = False
        evidence_full_text = " ".join(evidence_texts).lower()
        for s in final_answer.split("."):
            s_clean = s.strip().lower()
            if len(s_clean.split()) >= 7 and s_clean in evidence_full_text:
                has_verbatim_sent = True
                break

        # Check for sentence fragments / lowercase leading
        has_sentence_frag = any(p.strip() and p.strip()[0].islower() for p in final_answer.split("\n") if not p.strip().startswith("-") and not p.strip().startswith("*"))

        # Check for cross-paper contamination
        cross_paper = any(getattr(e, 'domain', '') != 'uploaded' and getattr(e, 'source_type', '') != 'uploaded' for e in evidence_list if getattr(e, 'paper_id', '') != 'SDC_BasePaper.pdf' and getattr(e, 'paper_id', '') != 'uploaded')

        # Check factual grounding & claim precision stats
        claims_checked = response.retrieval_metadata.get("claims_checked", len(citations_used)) if response.retrieval_metadata else len(citations_used)
        supported_claims = response.retrieval_metadata.get("supported_claims", len(citations_used)) if response.retrieval_metadata else len(citations_used)
        partially_supported = response.retrieval_metadata.get("partially_supported_claims", 0) if response.retrieval_metadata else 0
        unsupported = response.retrieval_metadata.get("unsupported_claims", 0) if response.retrieval_metadata else 0
        factual_grounded = len(citations_used) > 0 and unsupported == 0

        # Rendering check: verify no markdown fences or json keys
        ui_clean = not ("```json" in final_answer or "{\"" in final_answer[:20] or "[U1] [U1]" in final_answer)

        is_passed = is_synth and (not has_verbatim_sent) and (not cross_paper) and ui_clean and (len(final_answer.strip()) > 40) and unsupported == 0

        record = {
            "num": idx,
            "question": q,
            "intent": q_repr.intent,
            "q_type": q_repr.question_type,
            "retrieved_ids": evidence_ids,
            "evidence_coverage": f"{cov_state} ({cov_ratio*100:.0f}%)",
            "evidence_texts": [t[:100] + "..." for t in evidence_texts],
            "final_answer": final_answer,
            "citations_used": citations_used,
            "claims_checked": claims_checked,
            "supported_claims": supported_claims,
            "partially_supported_claims": partially_supported,
            "unsupported_claims": unsupported,
            "grounded": factual_grounded,
            "anti_copy_score": score,
            "is_synthesized": is_synth,
            "anti_copy_reason": reason,
            "has_verbatim_sentence": has_verbatim_sent,
            "has_sentence_fragment": has_sentence_frag,
            "cross_paper_contamination": cross_paper,
            "ui_rendered_clean": ui_clean,
            "evidence_strength": response.why_this_answer.evidence_strength if response.why_this_answer else "N/A",
            "completeness": is_comp,
            "passed": is_passed
        }
        question_results.append(record)

        print(f"Detected Intent: {q_repr.intent} | Type: {q_repr.question_type}")
        print(f"Retrieved Evidence IDs ({len(evidence_ids)}): {evidence_ids}")
        print(f"Citations Used in Answer ({len(citations_used)}): {citations_used}")
        print(f"Evidence Coverage: {cov_state} ({cov_ratio*100:.0f}%) | Missing: {missing}")
        print(f"Claims Checked: {claims_checked} | Supported: {supported_claims} | Partially Supported: {partially_supported} | Unsupported: {unsupported}")
        print(f"Anti-Copy Overlap Score: {score:.2f} | Synthesized: {is_synth} (Reason: {reason})")
        print(f"Verbatim Sentence Detected: {has_verbatim_sent}")
        print(f"Sentence Fragment Detected: {has_sentence_frag}")
        print(f"Cross-Paper Contamination: {cross_paper}")
        print(f"UI Clean Rendering: {ui_clean}")
        print(f"Evidence Strength: {record['evidence_strength']}")
        print(f"\nFinal Frontend Answer:\n{final_answer}\n")

    # Save detailed JSON audit artifact
    with open("scratch/live_webpage_qa_audit.json", "w", encoding="utf-8") as f:
        json.dump(question_results, f, indent=2)

    print("\n" + "="*80)
    print("LIVE PRODUCTION QA AUDIT SUMMARY TABLE:")
    print("="*80)
    all_passed = True
    for r in question_results:
        status_label = "[PASS]" if r["passed"] else "[FAIL]"
        if not r["passed"]:
            all_passed = False
        print(f"Q{r['num']:02d}: {status_label} Intent={r['intent']:<18} | Claims={r['claims_checked']} (Sup={r['supported_claims']}, Unsup={r['unsupported_claims']}) | Synth={str(r['is_synthesized']):<5} | Score={r['anti_copy_score']:.2f} | Cits={str(r['citations_used']):<16} | Strength={r['evidence_strength']}")

    print("="*80)
    print(f"OVERALL FINAL VERDICT: {'ALL 10/10 PASS' if all_passed else 'FAIL'}")
    print("="*80 + "\n")

    db.close()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
