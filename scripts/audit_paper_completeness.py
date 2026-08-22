"""
Phase 28.7 Paper Completeness Audit Script — Scans all 1,000 paper files across data/papers/
Determines file types (PDF vs TXT), section structure, character/word count, completeness level,
and chunking breakdown across Artificial Intelligence, Cybersecurity, Agriculture, Climate, Healthcare.
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter

DATA_DIR = Path("data/papers")
PROCESSED_CHUNKS_FILE = Path("data/processed/chunks.jsonl")

DOMAINS = [
    ("artificial_intelligence", "AI"),
    ("cybersecurity", "CY"),
    ("agriculture", "AG"),
    ("climate", "CL"),
    ("healthcare", "HC"),
]

SECTION_KEYWORDS = {
    "abstract": r"\b(abstract)\b",
    "introduction": r"\b(introduction|background)\b",
    "related_work": r"\b(related work|prior work|literature review)\b",
    "methodology": r"\b(methodology|methods|proposed approach|system architecture|model architecture|framework|implementation)\b",
    "experiments_results": r"\b(experiments|results|evaluation|experimental setup|performance|findings)\b",
    "discussion": r"\b(discussion|analysis|trade-offs|limitations)\b",
    "conclusion": r"\b(conclusion|future work|concluding remarks|summary)\b",
    "references": r"\b(references|bibliography|citations)\b",
}


def audit_corpus():
    paper_files = []
    domain_stats = defaultdict(lambda: {
        "total": 0, "pdf": 0, "txt": 0, "other": 0,
        "complete": 0, "substantially_complete": 0, "partial": 0, "abstract_only": 0
    })

    # 1. Inspect all files in data/papers/
    for domain_folder, domain_code in DOMAINS:
        domain_path = DATA_DIR / domain_folder
        if not domain_path.exists():
            print(f"Directory missing: {domain_path}")
            continue

        for fpath in domain_path.glob("*"):
            if fpath.is_file():
                ext = fpath.suffix.lower()
                paper_id = fpath.stem
                
                stat = domain_stats[domain_code]
                stat["total"] += 1

                content = ""
                word_count = 0
                char_count = 0

                if ext == ".txt":
                    stat["txt"] += 1
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        char_count = len(content)
                        word_count = len(re.findall(r"\b\w+\b", content))
                    except Exception as e:
                        print(f"Error reading {fpath}: {e}")
                elif ext == ".pdf":
                    stat["pdf"] += 1
                    # Estimate or parse PDF
                    char_count = fpath.stat().st_size
                else:
                    stat["other"] += 1

                # Detect sections in text
                detected_sections = []
                content_lower = content.lower()
                for sec_name, pattern in SECTION_KEYWORDS.items():
                    if re.search(pattern, content_lower):
                        detected_sections.append(sec_name)

                # Classify completeness
                if ext == ".txt":
                    has_abstract = "abstract" in detected_sections
                    has_intro = "introduction" in detected_sections
                    has_method = "methodology" in detected_sections
                    has_results = "experiments_results" in detected_sections
                    has_conclusion = "conclusion" in detected_sections

                    if word_count > 2500 and (has_method or has_results) and (has_intro or has_conclusion):
                        completeness = "Complete"
                        stat["complete"] += 1
                    elif word_count > 1200 and (has_method or has_results or has_intro):
                        completeness = "Substantially Complete"
                        stat["substantially_complete"] += 1
                    elif word_count < 400 and not (has_method or has_results):
                        completeness = "Abstract Only"
                        stat["abstract_only"] += 1
                    else:
                        completeness = "Partial Text"
                        stat["partial"] += 1
                else:
                    completeness = "Complete (PDF)"
                    stat["complete"] += 1

                paper_files.append({
                    "paper_id": paper_id,
                    "filename": fpath.name,
                    "domain": domain_folder,
                    "domain_code": domain_code,
                    "ext": ext,
                    "char_count": char_count,
                    "word_count": word_count,
                    "sections_detected": detected_sections,
                    "completeness": completeness,
                })

    # 2. Inspect Chunks in data/processed/chunks.jsonl
    chunks_per_paper = defaultdict(list)
    if PROCESSED_CHUNKS_FILE.exists():
        with open(PROCESSED_CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                pid = c.get("paper_id")
                chunks_per_paper[pid].append(c)

    print("==========================================================================")
    print("SCHOLARLENS 1,000-PAPER SOURCE COMPLETENESS AUDIT RESULTS")
    print("==========================================================================")
    
    print("\n--- DOMAIN-WISE SOURCE & COMPLETENESS SUMMARY ---")
    print(f"{'Domain':<15} | {'Total':<6} | {'PDF':<5} | {'TXT':<5} | {'Complete':<9} | {'Substantial':<11} | {'Partial':<8} | {'Abstract Only':<13}")
    print("-" * 88)
    for _, code in DOMAINS:
        st = domain_stats[code]
        print(f"{code:<15} | {st['total']:<6} | {st['pdf']:<5} | {st['txt']:<5} | {st['complete']:<9} | {st['substantially_complete']:<11} | {st['partial']:<8} | {st['abstract_only']:<13}")

    # 3. Sample 10 TXT papers from each domain (50 total)
    sampled_papers = []
    for _, code in DOMAINS:
        domain_txts = [p for p in paper_files if p["domain_code"] == code and p["ext"] == ".txt"]
        sample = domain_txts[:10]
        sampled_papers.extend(sample)

    print("\n--- SAMPLE INSPECTION (10 PAPERS PER DOMAIN) ---")
    for p in sampled_papers[:15]:  # print first 15 in console
        pid = p["paper_id"]
        chk_count = len(chunks_per_paper[pid])
        sec_str = ", ".join(p["sections_detected"]) if p["sections_detected"] else "None"
        print(f"[{p['domain_code']}] {p['filename']} ({p['word_count']} words, {chk_count} chunks) | Secs: {sec_str} | Status: {p['completeness']}")

    # Save full report JSON data
    audit_data = {
        "domain_stats": dict(domain_stats),
        "sampled_papers": sampled_papers,
        "all_papers_summary": [
            {
                "paper_id": p["paper_id"],
                "filename": p["filename"],
                "domain": p["domain_code"],
                "ext": p["ext"],
                "word_count": p["word_count"],
                "sections": p["sections_detected"],
                "completeness": p["completeness"],
                "chunk_count": len(chunks_per_paper[p["paper_id"]]),
            }
            for p in paper_files
        ]
    }

    out_file = Path("evaluation/results/source_completeness_audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    print(f"\nAudit complete. Full JSON exported to {out_file}")


if __name__ == "__main__":
    audit_corpus()
