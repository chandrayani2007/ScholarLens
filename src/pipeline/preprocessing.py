import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import fitz

logger = logging.getLogger(__name__)

# Regular expressions for common scientific heading variations
HEADING_PATTERNS = [
    # Numbered headings (e.g., 1. Introduction, 2.1 Related Work, I. Introduction)
    re.compile(r"^\s*(?:[IVXLCDM]+\.|\d+(?:\.\d+)*\.?)\s+([A-Z][a-zA-Z0-9\s\-,:()]+)$"),
    # Standard section names (case-insensitive)
    re.compile(
        r"^\s*(abstract|introduction|background|related\s+work|literature\s+review|"
        r"methodology|methods|materials\s+and\s+methods|proposed\s+(?:method|system|architecture|framework)|"
        r"system\s+model|experiments?|experimental\s+setup|evaluation|results?|"
        r"discussions?|conclusions?|conclusions?\s+and\s+future\s+work|future\s+work|"
        r"acknowledgements?|references?|bibliography|appendices|appendix)\s*$",
        re.IGNORECASE
    )
]

# Patterns for lines to remove (headers, footers, publisher watermarks, page numbers)
REMOVE_PATTERNS = [
    re.compile(r"^arXiv:\d{4}\.\d{5}v\d+\s+\[[a-z\-]+(\.[A-Z\-]+)?\]\s+\d+\s+[A-Za-z]+\s+\d{4}$", re.IGNORECASE), # arXiv watermark
    re.compile(r"^page\s+\d+\s+of\s+\d+$", re.IGNORECASE), # Page X of Y
    re.compile(r"^\d+\s*$", re.IGNORECASE), # Solo page numbers
    re.compile(r"^proceedings\s+of\s+.*$", re.IGNORECASE), # Publisher footer/header
    re.compile(r"^journal\s+of\s+.*$", re.IGNORECASE),
    re.compile(r"^ieee\s+transactions\s+on\s+.*$", re.IGNORECASE),
]

class PDFPreprocessingPipeline:
    """Orchestrates layout-aware PDF text extraction, cleaning, and section detection."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("data/metadata/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "sections.jsonl"

    def extract_layout_aware_text(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract blocks from PDF, sort them left-to-right/top-to-bottom for 2-column layouts.
        Returns a list of pages, where each page is a dict containing 'page_no' and 'blocks'.
        """
        doc = fitz.open(pdf_path)
        pages_data = []

        for page_idx, page in enumerate(doc):
            page_no = page_idx + 1
            # Retrieve text blocks
            raw_blocks = page.get_text("blocks")
            
            # Filter text blocks
            text_blocks = [b for b in raw_blocks if b[6] == 0] # type 0 is text
            
            # Sort layout aware
            sorted_blocks = self._sort_blocks(text_blocks, page.rect.width)
            
            pages_data.append({
                "page_no": page_no,
                "blocks": [b[4] for b in sorted_blocks]
            })

        doc.close()
        return pages_data

    def _sort_blocks(self, blocks: List[Tuple], page_width: float) -> List[Tuple]:
        """Sort blocks using column-aware spatial heuristics."""
        mid_x = page_width / 2.0
        
        left_col = []
        right_col = []
        full_width = []
        
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            if x1 <= mid_x + 10:
                left_col.append(b)
            elif x0 >= mid_x - 10:
                right_col.append(b)
            else:
                full_width.append(b)
                
        # Sort left column top-to-bottom
        left_col.sort(key=lambda b: b[1])
        # Sort right column top-to-bottom
        right_col.sort(key=lambda b: b[1])
        
        combined = []
        for b in left_col:
            combined.append((b, "left"))
        for b in right_col:
            combined.append((b, "right"))
        for b in full_width:
            combined.append((b, "full"))
            
        combined.sort(key=lambda item: item[0][1])
        
        final_blocks = []
        i = 0
        n = len(combined)
        while i < n:
            item, col_type = combined[i]
            if col_type == "full":
                final_blocks.append(item)
                i += 1
            else:
                region_left = []
                region_right = []
                
                current_y_limit = item[3]
                j = i
                while j < n and combined[j][1] != "full" and combined[j][0][1] < current_y_limit + 50:
                    current_y_limit = max(current_y_limit, combined[j][0][3])
                    if combined[j][1] == "left":
                        region_left.append(combined[j][0])
                    else:
                        region_right.append(combined[j][0])
                    j += 1
                
                region_left.sort(key=lambda b: b[1])
                region_right.sort(key=lambda b: b[1])
                
                final_blocks.extend(region_left)
                final_blocks.extend(region_right)
                
                i = max(j, i + 1)
                
        return final_blocks

    def clean_text_block(self, text: str) -> str:
        """Clean running headers/footers, metadata noise, and excessive whitespace."""
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            
            skip = False
            for p in REMOVE_PATTERNS:
                if p.match(stripped):
                    skip = True
                    break
            if skip:
                continue
                
            collapsed = re.sub(r"\s+", " ", stripped)
            cleaned_lines.append(collapsed)
            
        return " ".join(cleaned_lines)

    def detect_heading(self, text: str) -> Optional[str]:
        """Detect if a short text block is a section heading."""
        stripped = text.strip()
        if not stripped or len(stripped) > 80:
            return None
            
        for p in HEADING_PATTERNS:
            m = p.match(stripped)
            if m:
                if len(m.groups()) >= 1 and m.group(1):
                    return m.group(1).strip().title()
                return stripped.title()
        return None

    def segment_paper(self, paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pages, detect sections, clean text, and construct section list."""
        pdf_path = Path(paper["local_path"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        pages_data = self.extract_layout_aware_text(pdf_path)
        
        sections = []
        current_section = {
            "section_name": "Header",
            "page_start": 1,
            "page_end": 1,
            "text_blocks": []
        }
        
        section_order = 1
        
        for p_idx, page in enumerate(pages_data):
            page_no = page["page_no"]
            
            for block in page["blocks"]:
                cleaned_block = self.clean_text_block(block)
                if not cleaned_block:
                    continue
                    
                heading_name = self.detect_heading(block)
                if heading_name:
                    if current_section["text_blocks"]:
                        text_content = "\n\n".join(current_section["text_blocks"])
                        sections.append({
                            "section_id": f"SEC_{paper['paper_id']}_{section_order:03d}",
                            "paper_id": paper["paper_id"],
                            "domain": paper["domain"],
                            "subtopic": paper["subtopic"],
                            "section_name": current_section["section_name"],
                            "section_order": section_order,
                            "page_start": current_section["page_start"],
                            "page_end": current_section["page_end"],
                            "text": text_content
                        })
                        section_order += 1
                        
                    current_section = {
                        "section_name": heading_name,
                        "page_start": page_no,
                        "page_end": page_no,
                        "text_blocks": []
                    }
                else:
                    current_section["text_blocks"].append(cleaned_block)
                    current_section["page_end"] = page_no
                    
        if current_section["text_blocks"]:
            text_content = "\n\n".join(current_section["text_blocks"])
            sections.append({
                "section_id": f"SEC_{paper['paper_id']}_{section_order:03d}",
                "paper_id": paper["paper_id"],
                "domain": paper["domain"],
                "subtopic": paper["subtopic"],
                "section_name": current_section["section_name"],
                "section_order": section_order,
                "page_start": current_section["page_start"],
                "page_end": current_section["page_end"],
                "text": text_content
            })
            
        return sections
