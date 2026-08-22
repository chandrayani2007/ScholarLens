"""
Phase 28.7 PDF Index Rebuilder
Extracts text from all 1,000 genuine PDFs (200 AI, 200 CY, 200 AG, 200 CL, 200 HC),
parses sections, chunks text via tiktoken cl100k_base (~250 tokens per chunk),
rebuilds data/metadata/papers_metadata.json, data/processed/chunks.jsonl,
rebuilds ChromaDB collection 'research_mind_chunks', and rebuilds BM25 index 'bm25.pkl'.
"""

import json
import os
import re
import shutil
import fitz  # PyMuPDF
from pathlib import Path
from collections import defaultdict
from src.pipeline.chunking import split_into_sentences, TokenCounter
from src.pipeline.embedding import IndexingConfig, EmbeddingGenerator
from src.pipeline.chroma_indexer import ChromaDBIndexer
from src.pipeline.indexing import BM25Indexer

DATA_DIR = Path("data/papers")
METADATA_FILE = Path("data/metadata/papers_metadata.json")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/indexes")

DOMAINS = [
    ("artificial_intelligence", "AI"),
    ("cybersecurity", "CY"),
    ("agriculture", "AG"),
    ("climate", "CL"),
    ("healthcare", "HC"),
]

SECTION_KEYWORDS = [
    ("SEC_ABSTRACT", r"\b(abstract)\b"),
    ("SEC_INTRODUCTION", r"\b(introduction|background)\b"),
    ("SEC_RELATED_WORK", r"\b(related work|literature review)\b"),
    ("SEC_METHODOLOGY", r"\b(methodology|methods|proposed approach|system architecture|model)\b"),
    ("SEC_RESULTS", r"\b(experiments|results|evaluation|findings)\b"),
    ("SEC_DISCUSSION", r"\b(discussion|analysis|limitations)\b"),
    ("SEC_CONCLUSION", r"\b(conclusion|concluding remarks|summary)\b"),
]


def extract_pdf_sections(pdf_path: Path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        txt = page.get_text()
        if txt:
            full_text += txt.strip() + "\n\n"
    doc.close()

    lines = full_text.splitlines()
    sections = []
    current_sec_id = "SEC_MAIN"
    current_sec_name = "Main Content"
    current_sec_lines = []

    for line in lines:
        line_clean = line.strip().lower()
        matched_sec = None
        for sec_id, pattern in SECTION_KEYWORDS:
            if len(line_clean) < 60 and re.search(pattern, line_clean):
                matched_sec = (sec_id, line.strip())
                break
        
        if matched_sec:
            if current_sec_lines:
                sections.append({
                    "section_id": current_sec_id,
                    "section_name": current_sec_name,
                    "text": "\n".join(current_sec_lines).strip()
                })
            current_sec_id, current_sec_name = matched_sec
            current_sec_lines = [line]
        else:
            current_sec_lines.append(line)

    if current_sec_lines:
        sections.append({
            "section_id": current_sec_id,
            "section_name": current_sec_name,
            "text": "\n".join(current_sec_lines).strip()
        })

    return full_text, sections


def rebuild_corpus_indexes():
    print("==========================================================================")
    print("REBUILDING CORPUS INDEXES FROM 1,000 GENUINE PDFs")
    print("==========================================================================")

    token_counter = TokenCounter("cl100k_base")
    papers_metadata = {}
    all_chunks = []

    paper_count = 0
    chunk_count = 0

    for dom_folder, dom_code in DOMAINS:
        dom_dir = DATA_DIR / dom_folder
        print(f"\nProcessing PDF papers in {dom_folder} [{dom_code}001-{dom_code}200]...")

        for num in range(1, 201):
            paper_id = f"{dom_code}{num:03d}"
            pdf_path = dom_dir / f"{paper_id}.pdf"

            if not pdf_path.exists():
                print(f"ERROR: Missing PDF {pdf_path}!")
                continue

            full_text, sections = extract_pdf_sections(pdf_path)

            # Metadata entry
            first_line = full_text.splitlines()[0] if full_text.splitlines() else paper_id
            title = first_line[:120].strip() or f"Paper {paper_id}"

            papers_metadata[paper_id] = {
                "paper_id": paper_id,
                "title": title,
                "authors": ["Academic Researchers"],
                "published_date": "2024",
                "domain": dom_folder,
                "file_path": f"data/papers/{dom_folder}/{paper_id}.pdf",
                "source_type": "pdf"
            }

            paper_count += 1
            paper_chunk_seq = 1

            for sec in sections:
                sec_text = sec["text"]
                if not sec_text.strip():
                    continue

                sents = split_into_sentences(sec_text)
                cur_chunk_sents = []
                cur_tokens = 0

                for sent in sents:
                    stokens = token_counter.count(sent)
                    if cur_tokens + stokens > 250 and cur_chunk_sents:
                        chunk_id = f"{paper_id}_C{paper_chunk_seq:02d}"
                        unit_id = f"{paper_id}_U{paper_chunk_seq:02d}"
                        chunk_text = " ".join(cur_chunk_sents)

                        all_chunks.append({
                            "unit_id": unit_id,
                            "parent_chunk_id": chunk_id,
                            "chunk_id": chunk_id,
                            "paper_id": paper_id,
                            "section_id": sec["section_id"],
                            "section_name": sec["section_name"],
                            "domain": dom_folder,
                            "subtopic": "general",
                            "page_start": 1,
                            "page_end": 1,
                            "text": chunk_text
                        })

                        paper_chunk_seq += 1
                        chunk_count += 1
                        cur_chunk_sents = [sent]
                        cur_tokens = stokens
                    else:
                        cur_chunk_sents.append(sent)
                        cur_tokens += stokens

                if cur_chunk_sents:
                    chunk_id = f"{paper_id}_C{paper_chunk_seq:02d}"
                    unit_id = f"{paper_id}_U{paper_chunk_seq:02d}"
                    chunk_text = " ".join(cur_chunk_sents)

                    all_chunks.append({
                        "unit_id": unit_id,
                        "parent_chunk_id": chunk_id,
                        "chunk_id": chunk_id,
                        "paper_id": paper_id,
                        "section_id": sec["section_id"],
                        "section_name": sec["section_name"],
                        "domain": dom_folder,
                        "subtopic": "general",
                        "page_start": 1,
                        "page_end": 1,
                        "text": chunk_text
                    })

                    paper_chunk_seq += 1
                    chunk_count += 1

    # Save updated metadata file
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(papers_metadata, f, indent=2)
    print(f"\nSaved metadata for {len(papers_metadata)} papers to {METADATA_FILE}")

    # Save updated chunks file
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {CHUNKS_FILE}")

    # Generate Embeddings & Rebuild ChromaDB Index
    print("\nGenerating Embeddings & Rebuilding ChromaDB Collection 'research_mind_chunks'...")
    config = IndexingConfig()
    generator = EmbeddingGenerator(config)

    units = generator.prepare_units(all_chunks)
    embeddings = generator.encode_units(units, show_progress=True)

    chroma_db_dir = INDEX_DIR / "chroma"
    if chroma_db_dir.exists():
        shutil.rmtree(chroma_db_dir)

    chroma_indexer = ChromaDBIndexer(db_dir=chroma_db_dir)
    chroma_indexer.get_or_create_collection()
    chroma_indexer.build_index(embeddings, units)
    print(f"Successfully indexed {len(units)} units into ChromaDB.")

    # Rebuild BM25 Index
    print("\nRebuilding BM25 Index 'bm25.pkl'...")
    bm25_indexer = BM25Indexer()
    bm25_indexer.build_index(units)
    bm25_file = INDEX_DIR / "bm25.pkl"
    bm25_indexer.save(bm25_file)
    print(f"Successfully saved BM25 index to {bm25_file}")

    print("\n==========================================================================")
    print(f"INDEX REBUILD COMPLETE: {paper_count} PDFs, {chunk_count} chunks, {len(units)} vector units.")
    print("==========================================================================")


if __name__ == "__main__":
    rebuild_corpus_indexes()
