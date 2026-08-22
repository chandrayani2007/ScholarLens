"""
Phase 22 — Run 11 Required Benchmark Queries through Live Pipeline
"""

import sys
from src.pipeline.rag import RAGPipeline
from src.pipeline.llm import MockLLMProvider


def clean_str(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def main():
    pipeline = RAGPipeline(llm_provider=MockLLMProvider())

    benchmark_queries = [
        ("1. RAG Definition", "What is retrieval-augmented generation, and how does it improve large language model responses?", {"domain": "artificial_intelligence"}),
        ("2. RAG Evaluation Challenges", "What are the main challenges in evaluating retrieval-augmented generation systems?", {"domain": "artificial_intelligence"}),
        ("3. Cyber Attack Detection", "How can deep learning techniques be used to detect cyber attacks?", {"domain": "cybersecurity"}),
        ("4. Anomalous Traffic Detection", "How can machine learning help identify anomalous network traffic?", {"domain": "cybersecurity"}),
        ("5. IoT Smart Irrigation", "How can IoT improve smart irrigation systems?", {"domain": "agriculture"}),
        ("6. Soil Moisture Optimization", "How does soil moisture monitoring help optimize irrigation?", {"domain": "agriculture"}),
        ("7. Medical Image Diagnosis", "How is deep learning used for medical image analysis and diagnosis?", {"domain": "healthcare"}),
        ("8. Explainable AI in Healthcare", "What role does explainable AI play in healthcare applications?", {"domain": "healthcare"}),
        ("9. Climate Pattern Prediction", "How can machine learning be used to predict climate patterns?", {"domain": "climate"}),
        ("10. Extreme Weather Prediction", "How can machine learning improve extreme weather prediction?", {"domain": "climate"}),
        ("11. Out-of-Corpus Negative Query", "What is quantum teleportation using biological neural networks?", None),
    ]

    print("==========================================================================")
    print("PHASE 22 — 11 CANONICAL BENCHMARK PIPELINE VERIFICATION RESULTS")
    print("==========================================================================")

    for label, q, filters in benchmark_queries:
        res = pipeline.answer(q, filters=filters)
        print(f"\nQUERY: {label}")
        print(f"Question: {q}")
        print(f"Confidence: {res.confidence}")
        print(f"Evidence Strength: {res.why_this_answer.evidence_strength}")
        print(f"Confidence == Evidence Strength: {res.confidence == res.why_this_answer.evidence_strength}")
        print(f"Citations Count: {len(res.citations)} ({list(res.citations.keys())})")
        print(f"Key Sources: {res.why_this_answer.evidence_passages}")
        print(f"Answer: {res.answer}")
        print(f"Explanation: {clean_str(res.why_this_answer.explanation_summary)}")
        print("-" * 74)

if __name__ == "__main__":
    main()
