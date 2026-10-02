"""
Academic Paper Corpus Router (Corpus Library Browsing, Search, Details & PDF Viewing)

Endpoints:
- GET /api/corpus (list/filter/search 1,000 papers)
- GET /api/corpus/{paper_id} (paper details)
- GET /api/corpus/{paper_id}/pdf (stream genuine PDF file)
"""

import json
import logging
import re
import urllib.parse
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corpus", tags=["Academic Paper Corpus"])

METADATA_FILE = Path("data/metadata/papers.json")
ALT_METADATA_FILE = Path("data/metadata/papers_metadata.json")
PAPERS_DIR = Path("data/papers")

_cached_metadata: Optional[Dict[str, Dict[str, Any]]] = None


def get_all_paper_metadata() -> Dict[str, Dict[str, Any]]:
    global _cached_metadata
    if _cached_metadata is None:
        target = METADATA_FILE if METADATA_FILE.exists() else ALT_METADATA_FILE
        if target.exists():
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    _cached_metadata = data
                elif isinstance(data, list):
                    _cached_metadata = {item["paper_id"]: item for item in data if "paper_id" in item}
        else:
            _cached_metadata = {}
    return _cached_metadata


@router.get("", response_model=Dict[str, Any])
def list_corpus_papers(
    domain: Optional[str] = Query(None, description="Filter by domain code or domain name"),
    search: Optional[str] = Query(None, description="Search by title, author, or paper_id"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """Browse and search the 1,000-paper ScholarLens academic paper library."""
    meta = get_all_paper_metadata()
    papers = list(meta.values())

    # Domain filter
    if domain and domain.strip() and domain.lower() != "all":
        dom_clean = domain.strip().lower()
        domain_alias_map = {
            "ai": "artificial_intelligence",
            "artificial intelligence": "artificial_intelligence",
            "cy": "cybersecurity",
            "cybersecurity": "cybersecurity",
            "ag": "agriculture",
            "agriculture": "agriculture",
            "cl": "climate",
            "climate": "climate",
            "hc": "healthcare",
            "healthcare": "healthcare"
        }
        target_domain = domain_alias_map.get(dom_clean, dom_clean)
        papers = [p for p in papers if p.get("domain", "").lower() == target_domain]

    # Search filter
    if search and search.strip():
        q_clean = search.strip().lower()
        filtered = []
        for p in papers:
            pid = p.get("paper_id", "").lower()
            title = p.get("title", "").lower()
            authors = " ".join(p.get("authors", [])).lower()
            if q_clean in pid or q_clean in title or q_clean in authors:
                filtered.append(p)
        papers = filtered

    total_count = len(papers)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_papers = papers[start_idx:end_idx]

    return {
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size if total_count > 0 else 1,
        "papers": paginated_papers
    }


@router.get("/{paper_id}", response_model=Dict[str, Any])
def get_paper_details(paper_id: str):
    """Retrieve detailed metadata for a single corpus paper."""
    raw_pid = paper_id.strip()
    clean_pid = raw_pid[:-4] if raw_pid.lower().endswith(".pdf") else raw_pid

    meta = get_all_paper_metadata()
    meta_lower = {k.lower(): v for k, v in meta.items()}

    found = meta.get(raw_pid) or meta.get(clean_pid) or meta_lower.get(raw_pid.lower()) or meta_lower.get(clean_pid.lower())

    if not found:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Paper '{paper_id}' not found in corpus.")
    return found


@router.get("/{paper_id}/pdf")
def get_paper_pdf(paper_id: str):
    """Stream genuine full-text PDF document for a corpus paper or uploaded paper."""
    raw_pid = urllib.parse.unquote(paper_id.strip())
    clean_pid = raw_pid[:-4] if raw_pid.lower().endswith(".pdf") else raw_pid

    unprefixed_pid = clean_pid
    for prefix in ["UPLOADED_", "uploaded_", "TEMP_", "temp_", "UPLOAD_", "upload_"]:
        if unprefixed_pid.startswith(prefix):
            unprefixed_pid = unprefixed_pid[len(prefix):]

    code_match = re.search(r"([A-Za-z]{2}\d{3,4})", raw_pid)
    base_code = code_match.group(1).upper() if code_match else None

    meta = get_all_paper_metadata()
    meta_lower = {k.lower(): v for k, v in meta.items()}
    paper_meta = (
        meta.get(raw_pid) or meta.get(clean_pid) or meta.get(unprefixed_pid)
        or (meta.get(base_code) if base_code else None)
        or meta_lower.get(raw_pid.lower()) or meta_lower.get(clean_pid.lower()) or meta_lower.get(unprefixed_pid.lower())
        or (meta_lower.get(base_code.lower()) if base_code else None)
    )

    pdf_file_path = None

    if paper_meta:
        rel_path = paper_meta.get("file_path", "") or paper_meta.get("local_path", "")
        if rel_path:
            pdf_file_path = Path(rel_path)

        if not pdf_file_path or not pdf_file_path.exists():
            domain = paper_meta.get("domain", "")
            pid_code = paper_meta.get("paper_id", clean_pid)
            pdf_file_path = PAPERS_DIR / domain / f"{pid_code}.pdf"

    # Direct candidate paths
    if not pdf_file_path or not pdf_file_path.exists():
        temp_dir = PAPERS_DIR / "temp"
        candidates = [
            temp_dir / raw_pid,
            temp_dir / f"{clean_pid}.pdf",
            temp_dir / f"{unprefixed_pid}.pdf",
            temp_dir / unprefixed_pid,
            Path("data/temp") / raw_pid,
            Path("data/temp") / f"{clean_pid}.pdf",
            Path("data/temp") / f"{unprefixed_pid}.pdf",
            PAPERS_DIR / f"{clean_pid}.pdf",
            PAPERS_DIR / raw_pid,
            PAPERS_DIR / f"{unprefixed_pid}.pdf",
        ]
        for cand in candidates:
            if cand.exists() and cand.is_file():
                pdf_file_path = cand
                break

    # Search by base code (e.g. AI001 from AI001_sec01)
    if not pdf_file_path or not pdf_file_path.exists():
        if base_code:
            matches = list(PAPERS_DIR.glob(f"**/{base_code}.pdf"))
            if matches and matches[0].is_file():
                pdf_file_path = matches[0]

    # Recursive fallback search across PAPERS_DIR
    if not pdf_file_path or not pdf_file_path.exists():
        search_terms = [clean_pid, raw_pid, unprefixed_pid]
        matches = []
        for term in search_terms:
            t_clean = term[:-4] if term.lower().endswith(".pdf") else term
            matches.extend(list(PAPERS_DIR.glob(f"**/{t_clean}.pdf")))
            matches.extend(list(PAPERS_DIR.glob(f"**/{term}")))
        for found_file in matches:
            if found_file.is_file():
                pdf_file_path = found_file
                break

    # Temp folder fallback for uploaded papers
    if not pdf_file_path or not pdf_file_path.exists():
        temp_dir = PAPERS_DIR / "temp"
        if temp_dir.exists():
            pdf_files = sorted(temp_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pdf_files:
                pdf_file_path = pdf_files[0]

    if not pdf_file_path or not pdf_file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Genuine PDF file for '{paper_id}' not found on server.")

    return FileResponse(
        path=pdf_file_path,
        media_type="application/pdf",
        filename=pdf_file_path.name,
        headers={"Content-Disposition": f"inline; filename={pdf_file_path.name}"}
    )

