"""
Phase 2 — Chunk Validation Script

Comprehensive integrity checks for chunks.jsonl against papers.json and sections.jsonl.
"""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

import tiktoken


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    is_pilot = "--pilot" in sys.argv
    if is_pilot:
        CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks_pilot.jsonl"
    else:
        CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.jsonl"

    PAPERS_PATH = BASE_DIR / "data" / "metadata" / "papers.json"
    SECTIONS_PATH = BASE_DIR / "data" / "metadata" / "processed" / "sections.jsonl"
    CONFIG_PATH = BASE_DIR / "config" / "chunking.json"

    print(f"Validating: {CHUNKS_PATH}")
    print(f"Mode: {'PILOT' if is_pilot else 'FULL'}")

    # Load config
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    encoding = tiktoken.get_encoding(config["tokenizer"])

    # Load papers
    with open(PAPERS_PATH, "r", encoding="utf-8") as f:
        papers = json.load(f)
    paper_ids = {p["paper_id"] for p in papers}
    paper_map = {p["paper_id"]: p for p in papers}

    # Load sections
    sections = []
    with open(SECTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sections.append(json.loads(line))
    section_ids = {s["section_id"] for s in sections}
    section_map = {s["section_id"]: s for s in sections}

    # Load chunks
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print(f"\nLoaded {len(chunks)} chunks")

    results = []
    all_pass = True

    def check(name, condition, detail=""):
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        results.append((name, status, detail))
        icon = "[OK]" if condition else "[!!]"
        print(f"  {icon} [{status}] {name}" + (f" -- {detail}" if detail else ""))

    print("\n=== Integrity Checks ===")

    # 1. Unique chunk_ids
    all_chunk_ids = [c["chunk_id"] for c in chunks]
    unique_ids = len(set(all_chunk_ids))
    check("Unique chunk_ids", unique_ids == len(all_chunk_ids),
          f"{unique_ids} unique / {len(all_chunk_ids)} total")

    # 2. Valid paper_id
    invalid_paper_ids = [c["chunk_id"] for c in chunks if c["paper_id"] not in paper_ids]
    check("Valid paper_id references", len(invalid_paper_ids) == 0,
          f"{len(invalid_paper_ids)} invalid" if invalid_paper_ids else "all valid")

    # 3. Valid section_id
    invalid_section_ids = [c["chunk_id"] for c in chunks if c["section_id"] not in section_ids]
    check("Valid section_id references", len(invalid_section_ids) == 0,
          f"{len(invalid_section_ids)} invalid" if invalid_section_ids else "all valid")

    # 4. No empty chunks
    empty_chunks = [c["chunk_id"] for c in chunks if not c.get("text", "").strip()]
    check("No empty chunks", len(empty_chunks) == 0,
          f"{len(empty_chunks)} empty" if empty_chunks else "all non-empty")

    # 5. Token count accuracy (spot-check all)
    token_mismatches = []
    for c in chunks:
        actual = len(encoding.encode(c["text"]))
        if actual != c["token_count"]:
            token_mismatches.append((c["chunk_id"], c["token_count"], actual))
    check("Token count accuracy", len(token_mismatches) == 0,
          f"{len(token_mismatches)} mismatches" if token_mismatches else "all accurate")

    # 6. Page ordering
    bad_pages = [c["chunk_id"] for c in chunks if c["page_start"] > c["page_end"]]
    check("page_start <= page_end", len(bad_pages) == 0,
          f"{len(bad_pages)} violations" if bad_pages else "all valid")

    # 7. Chunk index validity — sequential per paper, no gaps
    paper_chunks = defaultdict(list)
    for c in chunks:
        paper_chunks[c["paper_id"]].append(c["chunk_index"])
    index_gaps = 0
    for pid, indices in paper_chunks.items():
        expected = list(range(1, len(indices) + 1))
        if sorted(indices) != expected:
            index_gaps += 1
    check("Chunk index continuity (per paper)", index_gaps == 0,
          f"{index_gaps} papers with gaps" if index_gaps else "all sequential")

    # 8. Domain/subtopic matches source section
    mismatches = []
    for c in chunks:
        sec = section_map.get(c["section_id"])
        if sec:
            if c["domain"] != sec["domain"] or c["subtopic"] != sec["subtopic"]:
                mismatches.append(c["chunk_id"])
    check("Domain/subtopic matches section", len(mismatches) == 0,
          f"{len(mismatches)} mismatches" if mismatches else "all match")

    # 9. Paper coverage
    if is_pilot:
        expected_papers = {"AI001", "CY001", "AG001", "HC001", "CL001"}
    else:
        expected_papers = paper_ids
    covered_papers = {c["paper_id"] for c in chunks}
    missing_papers = expected_papers - covered_papers
    check("Paper coverage", len(missing_papers) == 0,
          f"{len(covered_papers)}/{len(expected_papers)} papers" +
          (f", missing: {sorted(missing_papers)[:5]}" if missing_papers else ""))

    # 10. Section coverage — all sections with text produce at least one chunk
    if is_pilot:
        eligible_sections = {s["section_id"] for s in sections
                            if s["paper_id"] in expected_papers and s["text"].strip()}
    else:
        eligible_sections = {s["section_id"] for s in sections if s["text"].strip()}
    covered_sections = {c["section_id"] for c in chunks}
    missing_sections = eligible_sections - covered_sections
    check("Section coverage", len(missing_sections) == 0,
          f"{len(covered_sections)}/{len(eligible_sections)} sections" +
          (f", missing: {sorted(list(missing_sections))[:5]}" if missing_sections else ""))

    # 11. Required fields present
    required_fields = ["chunk_id", "paper_id", "section_id", "domain", "subtopic",
                       "section_name", "page_start", "page_end", "chunk_index",
                       "text", "token_count"]
    missing_fields = []
    for c in chunks:
        for field in required_fields:
            if field not in c:
                missing_fields.append((c.get("chunk_id", "UNKNOWN"), field))
    check("All required fields present", len(missing_fields) == 0,
          f"{len(missing_fields)} missing" if missing_fields else "all present")

    # Summary
    print(f"\n{'='*50}")
    passed = sum(1 for _, s, _ in results if s == "PASS")
    total = len(results)
    print(f"RESULT: {passed}/{total} checks passed")
    if all_pass:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
        for name, status, detail in results:
            if status == "FAIL":
                print(f"  FAILED: {name} -- {detail}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
