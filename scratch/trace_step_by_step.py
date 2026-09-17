import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath("."))

from src.pipeline.llm import get_llm_provider
from src.pipeline.rag import (
    RAGPipeline,
    GROUNDED_SYSTEM_PROMPT,
    AntiCopyValidator,
    ClaimGroundingValidator,
    CitationValidator,
    FinalSafetyGate,
    _chunk_uploaded_paper_text,
    GenericQuestionAnalyzer,
    EvidenceContextBuilder,
)
from scratch.verify_live_pipeline import OPEN_SCHOLAR_PAPER_TEXT

def main():
    q = "What is the main research problem addressed by this paper?"
    llm = get_llm_provider("gemini")

    print("=" * 80)
    print("STEP 1: TRACING REAL PRODUCTION PATH WITH GEMINI")
    print("=" * 80)

    q_repr = GenericQuestionAnalyzer.analyze(q)
    intent = q_repr.intent

    uploaded_passages = _chunk_uploaded_paper_text(
        OPEN_SCHOLAR_PAPER_TEXT, "SDC_BasePaper.pdf", q_repr=q_repr
    )

    context_text, evidence_items, citations_map = EvidenceContextBuilder.build_context(
        uploaded_passages,
        source_type="uploaded",
    )

    evidence_map = {e.citation_id: e for e in evidence_items}

    structure_hint = "### Research Problem\nDirectly synthesize the core research problem, challenges, or unmet needs addressed by the authors."

    user_prompt = (
        f"Research Question:\n{q}\n"
        f"Question Intent: {intent}\n\n"
        f"Retrieved Research Evidence Passages (UPLOADED):\n{context_text}\n\n"
        f"Instructions:\n"
        f"You are ScholarLens, an evidence-grounded scientific research assistant.\n\n"
        f"The passages below are SOURCE EVIDENCE ONLY. They are NOT the answer.\n"
        f"Answer the user's question by synthesizing information from the evidence.\n"
        f"Do NOT copy any retrieved passage verbatim.\n"
        f"Do NOT return a retrieved sentence as the answer.\n"
        f"Do NOT concatenate retrieved sentences.\n"
        f"Do NOT reproduce the evidence paragraph with a citation.\n"
        f"Rewrite and organize the supported information into a clear, natural research-level explanation.\n"
        f"For a research-problem question, identify the actual challenge, research gap, limitation, or unmet need described by the authors.\n"
        f"Do not mistake general background information for the research problem.\n"
        f"Use ONLY information supported by the evidence. Do not invent facts.\n"
        f"Every factual claim must be supported by an evidence citation like [U1], [U2].\n\n"
        f"Required Markdown Structure Outline:\n{structure_hint}\n\n"
        f"Return ONLY valid JSON matching this schema:\n"
        f"{{\n"
        f'  "answer": "### Research Problem\\n...",\n'
        f'  "confidence": "High" | "Moderate" | "Low" | "Insufficient",\n'
        f'  "evidence_strength": "High" | "Moderate" | "Weak" | "Insufficient",\n'
        f'  "limitations": "...",\n'
        f'  "why_this_answer": "..."\n'
        f"}}"
    )

    print("\n--- QUESTION ---")
    print(q)
    print("\n--- QUESTION INTENT ---")
    print(intent)
    print("\n--- RETRIEVED EVIDENCE ---")
    for ev in uploaded_passages[:5]:
        print(f"[{getattr(ev, 'citation_id', 'U1')}] Section: {getattr(ev, 'section_name', 'N/A')}\n  Text: {ev.text[:200]}...")

    print("\n--- GEMINI PROVIDER ---")
    print(f"Provider: {type(llm).__name__}, Model: {llm.model}")

    print("\n--- EXACT GEMINI PROMPT ---")
    print(user_prompt)

    print("\n" + "-" * 40 + " CALLING GEMINI API " + "-" * 40)
    raw_gemini_response = llm.generate(user_prompt, system_prompt=GROUNDED_SYSTEM_PROMPT)

    print("\n--- RAW GEMINI RESPONSE ---")
    print(raw_gemini_response)

    # Parsing
    pipeline = RAGPipeline(llm_provider=llm)
    parsed_answer, conf, conf_rat, lim, why = pipeline._parse_llm_output(raw_gemini_response)

    print("\n--- PARSED GEMINI ANSWER ---")
    print(parsed_answer)

    # Anti-copy check
    is_synth, ac_reason, overlap = AntiCopyValidator.validate_answer(parsed_answer, uploaded_passages)
    print(f"\n--- ANTI-COPY VALIDATION ---")
    print(f"Passed: {is_synth} (score={overlap:.2f}, reason: {ac_reason})")

    # Claim Grounding
    post_proc_answer, unsupp_count, active_tags = ClaimGroundingValidator.validate_and_filter_claims(
        parsed_answer, evidence_map, q
    )

    print("\n--- POST-PROCESSING ANSWER (After Claim Grounding) ---")
    print(post_proc_answer)

    # Final Safety Gate
    is_valid_cit, used_citations, invalid_citations = CitationValidator.validate(
        post_proc_answer, citations_map
    )
    active_tags = list(dict.fromkeys(used_citations + active_tags))
    active_citations = {t: citations_map[t] for t in active_tags if t in citations_map}
    final_answer, active_citations, _ = FinalSafetyGate.sanitize_response(
        post_proc_answer, active_citations, uploaded_passages, ["artificial_intelligence"], {"UPLOAD_"}
    )

    print("\n--- FINAL WEBPAGE ANSWER ---")
    print(final_answer)
    print("=" * 80)

if __name__ == "__main__":
    main()
