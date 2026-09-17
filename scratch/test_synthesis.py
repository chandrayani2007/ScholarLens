import re
import json

OPEN_SCHOLAR_EVIDENCE = """
[U1]
Paper: SDC_BasePaper.pdf
Section: Abstract / Introduction
Text:
Synthesizing knowledge from the scientific literature is essential for discovering new directions, refining methodologies and supporting evidence-based decisions, yet the rapid growth of publications makes it increasingly difficult for researchers to stay informed. Effective synthesis requires precise retrieval, accurate attribution and access to up-to-date literature. LLMs can assist but suffer from hallucinations, outdated pre-training data and limited attribution.

[U2]
Paper: SDC_BasePaper.pdf
Section: Methods / System Design
Text:
We introduce OpenScholar, an open scientific literature synthesis system. OpenScholar couples a high-capacity scientific retriever with a specialized 8B parameter language model and an iterative self-feedback inference loop (OSDS). The system retrieves relevant passages from a specialized scientific data store and dynamically critiques and refines draft responses.

[U3]
Paper: SDC_BasePaper.pdf
Section: Experiments / Benchmark
Text:
To evaluate literature synthesis, we present ScholarQABench, comprising Scholar-CS with 100 questions and Scholar-Multi with 108 questions and 250 expert-written answers across computer science, physics, biomedicine and neuroscience. The questions were authored by experienced PhD students and postdocs to reflect real-world literature review practices.

[U4]
Paper: SDC_BasePaper.pdf
Section: Results / Evaluation
Text:
Experimental results from the ScholarQABench computer science subset show that OpenScholar with our trained 8B or GPT-4o substantially outperforms other systems. OpenScholar-GPT-4o improves correctness by 12% over base GPT-4o. In human evaluations conducted by 16 domain experts, OpenScholar-8B and OpenScholar-GPT-4o responses were preferred over expert-written answers 51% and 70% of the time, respectively, compared with 32% for base GPT-4o.

[U5]
Paper: SDC_BasePaper.pdf
Section: Discussion / Limitations
Text:
On both single-paper and multi-paper tasks, non-retrieval augmented baselines struggle to generate correct citations and show limited coverage. Models without retrieval often produce fabricated citations. However, iterative feedback loops introduce additional computational latency during long-form synthesis.
"""

class AntiCopyValidator:
    @staticmethod
    def extract_evidence_sentences(evidence_text: str):
        all_sents = []
        cleaned = re.sub(r"\[[EOU]\d+\]\s*\nPaper:.*?\nSection:.*?\nText:\s*", "", evidence_text)
        cleaned = re.sub(r"\[\d+\]", "", cleaned).replace("\n", " ")
        sents = re.split(r"(?<=[.!?])\s+", cleaned)
        for s in sents:
            s_clean = s.strip().lower()
            if len(s_clean) > 20:
                all_sents.append(s_clean)
        return all_sents

    @staticmethod
    def validate_answer(answer: str, evidence_text: str):
        ev_sents = AntiCopyValidator.extract_evidence_sentences(evidence_text)
        ans_clean = re.sub(r"\[[EOU]\d+(?:\s*,\s*[EOU]\d+)*\]", "", answer)
        ans_lines = [l.strip() for l in ans_clean.split("\n") if l.strip() and not l.strip().startswith("#")]
        ans_sents = []
        for l in ans_lines:
            sents = re.split(r"(?<=[.!?])\s+", l)
            for s in sents:
                s_str = s.strip()
                if len(s_str) > 15:
                    ans_sents.append(s_str)

        for ans_s in ans_sents:
            ans_s_lower = ans_s.lower().rstrip(".!?,")
            ans_tokens = set(re.findall(r"\b[a-z]{3,}\b", ans_s_lower))
            if not ans_tokens:
                continue

            for ev_s in ev_sents:
                ev_s_clean = ev_s.rstrip(".!?,")
                # 1. Exact match check
                if ans_s_lower == ev_s_clean or (len(ans_s_lower) > 30 and ans_s_lower in ev_s_clean):
                    return False, f"Direct copy detected: answer sentence '{ans_s[:60]}...' matches evidence passage.", 1.0

                # 2. Long verbatim substring match (>= 8 words)
                words_ans = ans_s_lower.split()
                if len(words_ans) >= 8:
                    for i in range(len(words_ans) - 7):
                        phrase = " ".join(words_ans[i:i+8])
                        if phrase in ev_s_clean:
                            return False, f"Long verbatim substring detected: '{phrase}' in evidence.", 0.95

                # 3. High Jaccard token overlap check
                ev_tokens = set(re.findall(r"\b[a-z]{3,}\b", ev_s_clean))
                if ev_tokens:
                    intersection = ans_tokens.intersection(ev_tokens)
                    union = ans_tokens.union(ev_tokens)
                    jaccard = len(intersection) / len(union)
                    if jaccard >= 0.75:
                        return False, f"High token overlap ({jaccard:.2f} >= 0.75) with evidence sentence: '{ans_s}' vs '{ev_s}'.", jaccard

        return True, "Answer is properly synthesized.", 0.0

print("AntiCopyValidator loaded.")
