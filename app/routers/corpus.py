"""
Academic Paper Corpus Router (Corpus Library Browsing, Search, Details & PDF Viewing)

Endpoints:
- GET /api/corpus (list/filter/search 1,000 papers)
- GET /api/corpus/{paper_id} (paper details)
- GET /api/corpus/{paper_id}/pdf (stream genuine PDF file)
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/corpus", tags=["Academic Paper Corpus"])

METADATA_FILE = Path("data/metadata/papers_metadata.json")
PAPERS_DIR = Path("data/papers")

_cached_metadata: Optional[Dict[str, Dict[str, Any]]] = None


def get_all_paper_metadata() -> Dict[str, Dict[str, Any]]:
    global _cached_metadata
    if _cached_metadata is None:
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
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
    pid = paper_id.strip().upper()
    meta = get_all_paper_metadata()
    if pid not in meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Paper '{paper_id}' not found in corpus.")
    return meta[pid]


@router.get("/{paper_id}/pdf")
def get_paper_pdf(paper_id: str):
    """Stream genuine full-text PDF document for a corpus paper."""
    pid = paper_id.strip().upper()
    meta = get_all_paper_metadata()
    if pid not in meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Paper '{paper_id}' not found in metadata.")

    paper_meta = meta[pid]
    rel_path = paper_meta.get("file_path", "")
    pdf_file_path = Path(rel_path)

    if not pdf_file_path.exists():
        # Search by domain directory
        domain = paper_meta.get("domain", "")
        pdf_file_path = PAPERS_DIR / domain / f"{pid}.pdf"

    if not pdf_file_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Genuine PDF file for '{paper_id}' not found on server.")

    return FileResponse(
        path=pdf_file_path,
        media_type="application/pdf",
        filename=f"{pid}.pdf",
        headers={"Content-Disposition": f"inline; filename={pid}.pdf"}
    )
