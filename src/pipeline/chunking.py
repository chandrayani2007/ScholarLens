"""
Phase 2 — Chunk Generation Pipeline

Converts structured sections (sections.jsonl) into retrieval-ready chunks (chunks.jsonl).
Respects section → paragraph → sentence boundaries.
Uses tiktoken for deterministic token counting.
"""

import re
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import tiktoken

logger = logging.getLogger(__name__)


# Known abbreviations that should not trigger sentence splits
_ABBREVS = frozenset({
    "e.g", "i.e", "et al", "fig", "eq", "ref", "vol",
    "no", "vs", "dr", "mr", "mrs", "ms", "prof", "jr",
    "sr", "inc", "ltd", "corp", "approx", "dept", "est",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
})

# Simple sentence boundary: period/exclamation/question followed by space+uppercase or end
_SENT_BOUNDARY_RE = re.compile(r"([.!?])\s+")


@dataclass
class ChunkingConfig:
    """Configuration for the chunking pipeline."""
    chunk_size: int = 512
    chunk_overlap: int = 75
    min_chunk_size: int = 50
    tokenizer: str = "cl100k_base"

    @classmethod
    def from_file(cls, config_path: Path) -> "ChunkingConfig":
        """Load configuration from a JSON file."""
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TokenCounter:
    """Deterministic token counter using tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if not text:
            return 0
        return len(self.encoding.encode(text))


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences at .!? boundaries.
    Uses a post-split merge strategy to handle abbreviations rather than
    a lookbehind regex (which cannot handle variable-width patterns).
    """
    if not text.strip():
        return []

    # Split at every .!? followed by whitespace
    parts = _SENT_BOUNDARY_RE.split(text)

    # Re-assemble: parts alternate between text and delimiter
    # e.g. ["Hello", ".", "World", ".", "End"] for "Hello. World. End"
    raw_sentences = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] in ".!?":
            raw_sentences.append(parts[i] + parts[i + 1])
            i += 2
        else:
            raw_sentences.append(parts[i])
            i += 1

    # Merge back sentences that end with known abbreviations
    merged = []
    for sent in raw_sentences:
        sent = sent.strip()
        if not sent:
            continue

        if merged:
            # Check if previous sentence ends with an abbreviation
            prev = merged[-1]
            # Extract the last word before the period
            last_word_match = re.search(r"(\S+)\s*$", prev)
            if last_word_match:
                last_token = last_word_match.group(1).rstrip(".!?")
                if last_token.lower() in _ABBREVS:
                    # Merge with previous
                    merged[-1] = prev + " " + sent
                    continue
                # Also handle single uppercase letter followed by period (e.g., "J.")
                if re.match(r"^[A-Z]$", last_token):
                    merged[-1] = prev + " " + sent
                    continue

        merged.append(sent)

    return merged if merged else [text.strip()]


def split_into_paragraphs(text: str) -> List[str]:
    """Split section text into paragraphs on double-newline boundaries."""
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs


class ChunkingPipeline:
    """
    Converts sections into retrieval-ready chunks.

    Algorithm:
    1. Group sections by paper_id (preserving section_order)
    2. For each section:
       a. Split into paragraphs
       b. Accumulate paragraphs into chunks respecting chunk_size
       c. Split oversized paragraphs at sentence boundaries
       d. Apply overlap between consecutive chunks within a section
    3. Emit chunks with full provenance and deterministic IDs
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.token_counter = TokenCounter(config.tokenizer)

    def chunk_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all sections for a single paper (already grouped by paper_id).
        Returns a list of chunk records with sequential chunk_index.
        """
        all_chunks = []
        chunk_index = 1

        for section in sections:
            section_chunks = self._chunk_section(section)

            for chunk_text, page_start, page_end in section_chunks:
                token_count = self.token_counter.count(chunk_text)
                paper_id = section["paper_id"]

                all_chunks.append({
                    "chunk_id": f"{paper_id}_C{chunk_index:03d}",
                    "paper_id": paper_id,
                    "section_id": section["section_id"],
                    "domain": section["domain"],
                    "subtopic": section["subtopic"],
                    "section_name": section["section_name"],
                    "page_start": page_start,
                    "page_end": page_end,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "token_count": token_count,
                })
                chunk_index += 1

        return all_chunks

    def _chunk_section(self, section: Dict[str, Any]) -> List[Tuple[str, int, int]]:
        """
        Chunk a single section. Returns list of (text, page_start, page_end) tuples.
        """
        text = section.get("text", "")
        sec_page_start = section.get("page_start", 1)
        sec_page_end = section.get("page_end", 1)

        if not text.strip():
            return []

        paragraphs = split_into_paragraphs(text)
        if not paragraphs:
            return []

        # Build a list of text units with estimated page positions
        # We estimate page position proportionally within the section's page range
        text_units = self._build_text_units(paragraphs, sec_page_start, sec_page_end)

        # Accumulate units into chunks
        chunks = self._accumulate_chunks(text_units, sec_page_start, sec_page_end)

        return chunks

    def _build_text_units(
        self,
        paragraphs: List[str],
        sec_page_start: int,
        sec_page_end: int,
    ) -> List[Tuple[str, int]]:
        """
        Convert paragraphs into text units (paragraph or sentence level).
        Each unit is (text, estimated_page_number).

        If a paragraph exceeds chunk_size, split it into sentences.
        """
        total_chars = sum(len(p) for p in paragraphs)
        page_span = sec_page_end - sec_page_start + 1

        units = []
        cumulative_chars = 0

        for para in paragraphs:
            # Estimate page position proportionally
            if total_chars > 0 and page_span > 1:
                frac = cumulative_chars / total_chars
                est_page = sec_page_start + int(frac * (page_span - 1) + 0.5)
                est_page = min(est_page, sec_page_end)
            else:
                est_page = sec_page_start

            para_tokens = self.token_counter.count(para)

            if para_tokens <= self.config.chunk_size:
                units.append((para, est_page))
            else:
                # Oversized paragraph — split into sentences
                sentences = split_into_sentences(para)
                sent_cumulative = 0
                for sent in sentences:
                    # Estimate page within paragraph
                    if len(para) > 0 and page_span > 1:
                        sent_frac = (cumulative_chars + sent_cumulative) / total_chars
                        sent_page = sec_page_start + int(sent_frac * (page_span - 1) + 0.5)
                        sent_page = min(sent_page, sec_page_end)
                    else:
                        sent_page = est_page
                    units.append((sent, sent_page))
                    sent_cumulative += len(sent)

            cumulative_chars += len(para)

        return units

    def _accumulate_chunks(
        self,
        text_units: List[Tuple[str, int]],
        sec_page_start: int,
        sec_page_end: int,
    ) -> List[Tuple[str, int, int]]:
        """
        Accumulate text units into chunks, respecting chunk_size and applying overlap.
        Returns list of (chunk_text, page_start, page_end).
        """
        if not text_units:
            return []

        chunks: List[Tuple[str, int, int]] = []

        # Current chunk state
        current_texts: List[str] = []
        current_tokens: int = 0
        current_page_start: int = text_units[0][1]
        current_page_end: int = text_units[0][1]

        # Previous chunk's tail units for overlap
        prev_tail_texts: List[str] = []
        prev_tail_tokens: int = 0

        for unit_text, unit_page in text_units:
            unit_tokens = self.token_counter.count(unit_text)

            # Check if adding this unit would exceed chunk_size
            if current_texts and (current_tokens + unit_tokens) > self.config.chunk_size:
                # Emit current chunk
                chunk_text = "\n\n".join(current_texts)
                chunks.append((chunk_text, current_page_start, current_page_end))

                # Compute overlap from tail of current chunk
                prev_tail_texts, prev_tail_tokens = self._compute_overlap_tail(current_texts)

                # Start new chunk with overlap
                if prev_tail_texts:
                    current_texts = list(prev_tail_texts)
                    current_tokens = prev_tail_tokens
                    # Page start for overlap chunk inherits from the overlap content
                    current_page_start = unit_page
                else:
                    current_texts = []
                    current_tokens = 0
                    current_page_start = unit_page

                current_page_end = unit_page

            # Add unit to current chunk
            if not current_texts:
                current_page_start = unit_page

            current_texts.append(unit_text)
            current_tokens += unit_tokens
            current_page_end = max(current_page_end, unit_page)

        # Emit remaining content
        if current_texts:
            remaining_tokens = self.token_counter.count("\n\n".join(current_texts))

            if remaining_tokens < self.config.min_chunk_size and chunks:
                # Merge into previous chunk if too small
                prev_text, prev_ps, prev_pe = chunks[-1]
                merged_text = prev_text + "\n\n" + "\n\n".join(current_texts)
                chunks[-1] = (merged_text, prev_ps, max(prev_pe, current_page_end))
            else:
                # Emit as final chunk (even if small, if it's the only/first chunk)
                chunk_text = "\n\n".join(current_texts)
                chunks.append((chunk_text, current_page_start, current_page_end))

        return chunks

    def _compute_overlap_tail(self, texts: List[str]) -> Tuple[List[str], int]:
        """
        Extract the tail of texts that fits within chunk_overlap tokens.
        Used for creating overlap between consecutive chunks.
        """
        if not texts or self.config.chunk_overlap <= 0:
            return [], 0

        tail_texts = []
        tail_tokens = 0

        for text in reversed(texts):
            t = self.token_counter.count(text)
            if tail_tokens + t > self.config.chunk_overlap:
                break
            tail_texts.insert(0, text)
            tail_tokens += t

        return tail_texts, tail_tokens
