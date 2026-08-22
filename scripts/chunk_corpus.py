"""
Phase 2 — Chunk Corpus Runner

Reads sections.jsonl, runs the chunking pipeline, and writes chunks.jsonl.
Supports --pilot flag for 5-paper pilot run.
"""

import json
import sys
import logging
from pathlib import Path
from collections import Counter, defaultdict

from src.pipeline.chunking import ChunkingConfig, ChunkingPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PILOT_PAPER_IDS = {"AI001", "CY001", "AG001", "HC001", "CL001"}


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    CONFIG_PATH = BASE_DIR / "config" / "chunking.json"
    SECTIONS_PATH = BASE_DIR / "data" / "metadata" / "processed" / "sections.jsonl"
    OUTPUT_DIR = BASE_DIR / "data" / "processed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    is_pilot = "--pilot" in sys.argv

    if is_pilot:
        CHUNKS_PATH = OUTPUT_DIR / "chunks_pilot.jsonl"
        STATS_PATH = OUTPUT_DIR / "chunking_stats_pilot.json"
        print("=== PILOT MODE: Processing 5 representative papers ===")
        print(f"Papers: {sorted(PILOT_PAPER_IDS)}")
    else:
        CHUNKS_PATH = OUTPUT_DIR / "chunks.jsonl"
        STATS_PATH = OUTPUT_DIR / "chunking_stats.json"
        print("=== FULL MODE: Processing all papers ===")

    # Load config
    config = ChunkingConfig.from_file(CONFIG_PATH)
    print(f"\nChunking Configuration:")
    print(f"  chunk_size: {config.chunk_size} tokens")
    print(f"  chunk_overlap: {config.chunk_overlap} tokens")
    print(f"  min_chunk_size: {config.min_chunk_size} tokens")
    print(f"  tokenizer: {config.tokenizer}")

    # Load sections
    sections = []
    with open(SECTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sections.append(json.loads(line))

    print(f"\nLoaded {len(sections)} sections")

    # Group sections by paper_id, preserving section_order
    paper_sections = defaultdict(list)
    for s in sections:
        paper_sections[s["paper_id"]].append(s)

    # Sort each paper's sections by section_order
    for pid in paper_sections:
        paper_sections[pid].sort(key=lambda s: s["section_order"])

    # Filter for pilot if needed
    if is_pilot:
        paper_sections = {
            pid: secs
            for pid, secs in paper_sections.items()
            if pid in PILOT_PAPER_IDS
        }
        print(f"Pilot: {len(paper_sections)} papers selected")

    # Run chunking pipeline
    pipeline = ChunkingPipeline(config)

    total_papers = len(paper_sections)
    all_chunks = []
    paper_chunk_counts = {}
    domain_chunk_counts = Counter()
    subtopic_chunk_counts = Counter()

    sorted_paper_ids = sorted(paper_sections.keys())

    for idx, paper_id in enumerate(sorted_paper_ids, 1):
        secs = paper_sections[paper_id]
        domain = secs[0]["domain"]
        subtopic = secs[0]["subtopic"]

        print(f"[{idx}/{total_papers}] Chunking {paper_id} ({len(secs)} sections)...")

        try:
            chunks = pipeline.chunk_sections(secs)
            all_chunks.extend(chunks)
            paper_chunk_counts[paper_id] = len(chunks)
            domain_chunk_counts[domain] += len(chunks)
            subtopic_chunk_counts[subtopic] += len(chunks)
        except Exception as e:
            logger.error(f"Failed to chunk {paper_id}: {e}")
            paper_chunk_counts[paper_id] = 0

    # Write chunks.jsonl
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_chunks)} chunks to {CHUNKS_PATH}")

    # Compute statistics
    token_counts = [c["token_count"] for c in all_chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    median_tokens = sorted(token_counts)[len(token_counts) // 2] if token_counts else 0

    chunk_counts_list = list(paper_chunk_counts.values())
    avg_chunks_per_paper = sum(chunk_counts_list) / len(chunk_counts_list) if chunk_counts_list else 0

    # Token distribution buckets
    buckets = Counter()
    for tc in token_counts:
        if tc < 50:
            buckets["<50"] += 1
        elif tc < 100:
            buckets["50-99"] += 1
        elif tc < 200:
            buckets["100-199"] += 1
        elif tc < 300:
            buckets["200-299"] += 1
        elif tc < 400:
            buckets["300-399"] += 1
        elif tc < 500:
            buckets["400-499"] += 1
        elif tc < 600:
            buckets["500-599"] += 1
        elif tc < 700:
            buckets["600-699"] += 1
        else:
            buckets["700+"] += 1

    # Find extremely small and large chunks
    small_chunks = [(c["chunk_id"], c["token_count"], c["section_name"]) for c in all_chunks if c["token_count"] < config.min_chunk_size]
    large_chunks = [(c["chunk_id"], c["token_count"], c["section_name"]) for c in all_chunks if c["token_count"] > config.chunk_size * 1.5]

    # Papers with fewest/most chunks
    sorted_by_chunks = sorted(paper_chunk_counts.items(), key=lambda x: x[1])
    fewest_chunks = sorted_by_chunks[:5]
    most_chunks = sorted_by_chunks[-5:]

    stats = {
        "mode": "pilot" if is_pilot else "full",
        "config": config.to_dict(),
        "total_papers": total_papers,
        "total_sections_processed": sum(len(secs) for secs in paper_sections.values()),
        "total_chunks": len(all_chunks),
        "average_chunks_per_paper": round(avg_chunks_per_paper, 1),
        "token_stats": {
            "average": round(avg_tokens, 1),
            "median": median_tokens,
            "min": min_tokens,
            "max": max_tokens,
        },
        "token_distribution": dict(sorted(buckets.items())),
        "domain_chunk_counts": dict(sorted(domain_chunk_counts.items())),
        "subtopic_chunk_counts": dict(sorted(subtopic_chunk_counts.items())),
        "small_chunks_count": len(small_chunks),
        "large_chunks_count": len(large_chunks),
        "papers_fewest_chunks": fewest_chunks,
        "papers_most_chunks": most_chunks,
    }

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"CHUNKING EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Mode: {'PILOT' if is_pilot else 'FULL'}")
    print(f"Papers processed: {total_papers}")
    print(f"Sections processed: {stats['total_sections_processed']}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Average chunks/paper: {avg_chunks_per_paper:.1f}")
    print(f"\nToken Statistics:")
    print(f"  Average: {avg_tokens:.1f}")
    print(f"  Median: {median_tokens}")
    print(f"  Min: {min_tokens}")
    print(f"  Max: {max_tokens}")
    print(f"\nToken Distribution:")
    for bucket in ["<50", "50-99", "100-199", "200-299", "300-399", "400-499", "500-599", "600-699", "700+"]:
        if bucket in buckets:
            print(f"  {bucket}: {buckets[bucket]}")
    print(f"\nSmall chunks (<{config.min_chunk_size} tokens): {len(small_chunks)}")
    if small_chunks[:5]:
        for cid, tc, sn in small_chunks[:5]:
            print(f"  {cid}: {tc} tokens ({sn})")
    print(f"Large chunks (>{config.chunk_size * 1.5:.0f} tokens): {len(large_chunks)}")
    if large_chunks[:5]:
        for cid, tc, sn in large_chunks[:5]:
            print(f"  {cid}: {tc} tokens ({sn})")

    if is_pilot:
        print(f"\n{'='*60}")
        print(f"PILOT SAMPLE CHUNKS")
        print(f"{'='*60}")
        for paper_id in sorted(PILOT_PAPER_IDS):
            paper_chunks = [c for c in all_chunks if c["paper_id"] == paper_id]
            if paper_chunks:
                print(f"\n--- {paper_id}: {len(paper_chunks)} chunks ---")
                # Show first and last chunk
                for label, chunk in [("First", paper_chunks[0]), ("Last", paper_chunks[-1])]:
                    print(f"  [{label}] {chunk['chunk_id']}:")
                    print(f"    section: {chunk['section_name']} ({chunk['section_id']})")
                    print(f"    pages: {chunk['page_start']}-{chunk['page_end']}")
                    print(f"    tokens: {chunk['token_count']}")
                    preview = chunk['text'][:150].encode('ascii', errors='replace').decode('ascii')
                    print(f"    text preview: {preview}...")

    print(f"\nDomain breakdown:")
    for domain, count in sorted(domain_chunk_counts.items()):
        print(f"  {domain}: {count}")

    print(f"\nStats written to: {STATS_PATH}")


if __name__ == "__main__":
    main()
