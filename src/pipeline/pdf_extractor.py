"""
ScholarLens PDF Extractor Module
Uses PyMuPDF (fitz) for layout-aware PDF text extraction, section detection, and cleaning.
"""

import base64
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import fitz

logger = logging.getLogger(__name__)

HEADING_PATTERNS = [
    re.compile(r"^\s*(?:[IVXLCDM]+\.|\d+(?:\.\d+)*\.?)\s+([A-Z][a-zA-Z0-9\s\-,:()]+)$"),
    re.compile(
        r"^\s*(abstract|introduction|background|related\s+work|literature\s+review|"
        r"methodology|methods|materials\s+and\s+methods|proposed\s+(?:method|system|architecture|framework)|"
        r"system\s+model|experiments?|experimental\s+setup|evaluation|results?|"
        r"discussions?|conclusions?|conclusions?\s+and\s+future\s+work|future\s+work|"
        r"acknowledgements?|references?|bibliography|appendices|appendix)\s*$",
        re.IGNORECASE
    )
]

REMOVE_PATTERNS = [
    re.compile(r"^arXiv:\d{4}\.\d{5}v\d+\s+\[[a-z\-]+(\.[A-Z\-]+)?\]\s+\d+\s+[A-Za-z]+\s+\d{4}$", re.IGNORECASE),
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE),
    re.compile(r"^\d+\s*$", re.IGNORECASE),
    re.compile(r"^proceedings\s+of\s+.*$", re.IGNORECASE),
    re.compile(r"^journal\s+of\s+.*$", re.IGNORECASE),
    re.compile(r"^ieee\s+transactions\s+on\s+.*$", re.IGNORECASE),
]


@dataclass
class PDFExtractionResult:
    filename: str
    pages_count: int
    char_count: int
    text: str
    section_headings: List[str] = field(default_factory=list)
    first_chunk_preview: str = ""
    chunks_count: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PDFExtractor:
    """Robust PDF text extraction and layout parsing engine using PyMuPDF (fitz)."""

    @staticmethod
    def extract_from_bytes(pdf_bytes: bytes, filename: str = "uploaded_paper.pdf") -> PDFExtractionResult:
        if not pdf_bytes:
            return PDFExtractionResult(
                filename=filename, pages_count=0, char_count=0, text="", error="PDF byte stream is empty"
            )

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            logger.error(f"[PDF EXTRACTION ERROR] Failed to open PDF '{filename}': {e}")
            return PDFExtractionResult(
                filename=filename, pages_count=0, char_count=0, text="", error=f"Invalid or corrupted PDF file: {e}"
            )

        pages_count = len(doc)
        full_text_blocks = []
        section_headings = []

        for page_idx, page in enumerate(doc):
            page_no = page_idx + 1
            raw_blocks = page.get_text("blocks")
            # type 0 is text block
            text_blocks = [b for b in raw_blocks if b[6] == 0]

            # Sort blocks top-to-bottom, left-to-right for double-column layouts
            sorted_blocks = sorted(text_blocks, key=lambda b: (round(b[1] / 30) * 30, b[0]))

            page_lines = []
            for b in sorted_blocks:
                block_text = b[4].strip()
                if not block_text:
                    continue
                # Check line removal patterns
                lines = block_text.split("\n")
                clean_lines = []
                for l in lines:
                    l_str = l.strip()
                    if not l_str:
                        continue
                    if any(pat.match(l_str) for pat in REMOVE_PATTERNS):
                        continue
                    clean_lines.append(l_str)

                    # Heading detection
                    if len(l_str) < 80:
                        for h_pat in HEADING_PATTERNS:
                            if h_pat.match(l_str):
                                section_headings.append(f"P{page_no}: {l_str}")
                                break

                if clean_lines:
                    page_lines.append("\n".join(clean_lines))

            if page_lines:
                full_text_blocks.append(f"\n--- Page {page_no} ---\n" + "\n\n".join(page_lines))

        full_text = "\n\n".join(full_text_blocks).strip()
        char_count = len(full_text)

        # Rough chunk estimation
        chunks_count = (char_count // 600) + (1 if char_count % 600 > 0 else 0)
        first_chunk_preview = full_text[:300] if full_text else ""

        # Remove duplicate heading labels
        unique_headings = list(dict.fromkeys(section_headings))

        logger.info(
            f"[PDF EXTRACTION LOG] filename='{filename}' pages={pages_count} "
            f"chars={char_count} headings={len(unique_headings)} chunks={chunks_count}"
        )
        if unique_headings:
            logger.info(f"[PDF HEADINGS DETECTED] {unique_headings[:10]}")

        return PDFExtractionResult(
            filename=filename,
            pages_count=pages_count,
            char_count=char_count,
            text=full_text,
            section_headings=unique_headings,
            first_chunk_preview=first_chunk_preview,
            chunks_count=chunks_count,
        )

    @staticmethod
    def extract_from_base64(base64_str: str, filename: str = "uploaded_paper.pdf") -> PDFExtractionResult:
        try:
            # Strip data URI header if present (e.g. "data:application/pdf;base64,...")
            if "," in base64_str:
                base64_str = base64_str.split(",", 1)[1]
            pdf_bytes = base64.b64decode(base64_str)
            return PDFExtractor.extract_from_bytes(pdf_bytes, filename=filename)
        except Exception as e:
            logger.error(f"[PDF BASE64 DECODE ERROR] {e}")
            return PDFExtractionResult(
                filename=filename, pages_count=0, char_count=0, text="", error=f"Invalid base64 encoding: {e}"
            )

    @staticmethod
    def extract_from_file_path(path_str: str) -> PDFExtractionResult:
        try:
            with open(path_str, "rb") as f:
                content = f.read()
            filename = path_str.replace("\\", "/").split("/")[-1]
            return PDFExtractor.extract_from_bytes(content, filename=filename)
        except Exception as e:
            logger.error(f"[PDF FILE PATH READ ERROR] {e}")
            return PDFExtractionResult(
                filename=path_str, pages_count=0, char_count=0, text="", error=f"Failed to read file path: {e}"
            )
