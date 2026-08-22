"""
Phase 6 — Functional RAG Pilot Runner Script

Executes the evidence-grounded RAG pipeline on 5 pilot questions across AI, Cybersecurity, Agriculture, Healthcare, and Climate.

For each question reports:
- Question
- Domain / Subtopic
- Top Retrieved Evidence Passages
- Generated Evidence-Grounded Answer
- Validated Citations Mapping Table
- "Why This Answer?" Explainability Summary
- Limitations
"""

import json
import logging
from pathlib import Path

from src.pipeline.rag import RAGPipeline, RAGResponse

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PILOT_QUESTIONS = [
    {
        "domain": "artificial_intelligence",
        "subtopic": "retrieval_augmented_generation",
        "question": "What is retrieval augmented generation?",
    },
    {
        "domain": "cybersecurity",
        "subtopic": "network_security",
        "question": "How are network attacks detected?",
    },
    {
        "domain": "agriculture",
        "subtopic": "smart_farming",
        "question": "How is IoT used in smart farming?",
    },
    {
        "domain": "healthcare",
        "subtopic": "medical_imaging",
        "question": "How does medical image analysis support diagnosis?",
    },
    {
        "domain": "climate",
        "subtopic": "climate_modeling",
        "question": "How is machine learning used for climate prediction?",
    },
]


def main():
    print("=" * 80)
    print("PHASE 6 — EVIDENCE-GROUNDED RAG FUNCTIONAL PILOT")
    print("=" * 80)

    pipeline = RAGPipeline()
    pilot_outputs = []

    for idx, item in enumerate(PILOT_QUESTIONS, 1):
        domain = item["domain"]
        subtopic = item["subtopic"]
        q = item["question"]

        print(f"\n[{idx}/5] PILOT QUESTION ({domain.upper()} / {subtopic}):")
        print(f"Question: \"{q}\"")

        res: RAGResponse = pipeline.answer(q, filters={"domain": domain})
        pilot_outputs.append(res.to_dict())

        print("-" * 80)
        print(f"GENERATED ANSWER:\n{res.answer}")
        print("-" * 80)
        print(f"CONFIDENCE: {res.confidence} ({res.confidence_rationale})")
        print(f"LIMITATIONS: {res.limitations}")
        print("-" * 80)
        print("CITATIONS MAPPING:")
        for tag, cit in res.citations.items():
            paper_id = cit.paper_id if hasattr(cit, "paper_id") else cit["paper_id"]
            sec_name = cit.section_name if hasattr(cit, "section_name") else cit["section_name"]
            pages = cit.pages if hasattr(cit, "pages") else cit["pages"]
            chunk_id = cit.chunk_id if hasattr(cit, "chunk_id") else cit["chunk_id"]
            print(f"  {tag} -> Paper: {paper_id}, Section: \"{sec_name}\", Pages: {pages}, Chunk: {chunk_id}")
        print("-" * 80)
        print("WHY THIS ANSWER?:")
        w = res.why_this_answer
        print(f"  Contributing Papers: {w.contributing_papers}")
        print(f"  Contributing Sections: {w.contributing_sections}")
        print(f"  Evidence Strength: {w.evidence_strength}")
        print(f"  Summary: {w.explanation_summary}")
        print("=" * 80)

    # Save pilot outputs JSON artifact
    BASE_DIR = Path(__file__).resolve().parent.parent
    out_path = BASE_DIR / "evaluation" / "rag_pilot_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pilot_outputs, f, indent=2)

    logger.info(f"Pilot execution results saved to {out_path}")


if __name__ == "__main__":
    main()
