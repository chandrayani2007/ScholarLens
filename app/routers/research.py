"""
Research Query Router

Endpoints:
- POST /research/query
"""

import uuid
import hashlib
import logging
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session

from app.db.models import User
from app.dependencies import get_db, get_current_user, get_research_service
from app.schemas.query import ResearchQueryRequest
from app.schemas.response import ResearchQueryResponse
from app.services.research_service import ResearchService
from src.pipeline.pdf_extractor import PDFExtractor, PDFExtractionResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research API"])

ALLOWED_DOMAINS = {"artificial_intelligence", "cybersecurity", "agriculture", "healthcare", "climate"}


@router.post("/upload-paper")
async def upload_paper(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    """
    Upload an academic paper PDF or text document.
    Parses layout-aware text, detects headings, and returns extraction metadata.
    """
    filename = file.filename or "uploaded_paper.pdf"
    contents = await file.read()

    logger.info(f"[API UPLOAD PAPER] Received file '{filename}' ({len(contents)} bytes) from user_id={current_user.id}")

    if filename.lower().endswith(".pdf") or (file.content_type and "pdf" in file.content_type.lower()) or contents.startswith(b"%PDF-"):
        res = PDFExtractor.extract_from_bytes(contents, filename=filename)
    else:
        try:
            text = contents.decode("utf-8")
        except UnicodeDecodeError:
            text = contents.decode("latin-1", errors="replace")
        char_count = len(text)
        chunks_count = (char_count // 600) + 1
        res = PDFExtractionResult(
            filename=filename,
            pages_count=1,
            char_count=char_count,
            text=text,
            section_headings=["Uploaded Text"],
            first_chunk_preview=text[:300],
            chunks_count=chunks_count,
        )

    if res.error or not res.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=res.error or "Extracted paper text is empty or unreadable.",
        )

    return res.to_dict()


@router.post("/query", response_model=ResearchQueryResponse)
def execute_research_query(
    req: ResearchQueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    research_service: ResearchService = Depends(get_research_service),
):
    """
    Execute evidence-grounded research query over the 250-paper corpus via HybridRetriever and RAGPipeline.
    """
    request_id = f"REQ-{uuid.uuid4().hex[:8]}"
    q_hash = hashlib.sha256(req.question.encode("utf-8")).hexdigest()[:8]

    logger.info(f"[RESEARCH REQUEST] request_id={request_id} question_hash={q_hash} user_id={current_user.id} domain={req.domain}")

    if req.domain and req.domain.lower() not in ALLOWED_DOMAINS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid domain '{req.domain}'. Allowed domains: {sorted(ALLOWED_DOMAINS)}",
        )

    try:
        response = research_service.process_query(db=db, user_id=current_user.id, req=req, request_id=request_id, question_hash=q_hash)
        response.request_id = request_id
        response.question_hash = q_hash
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"[RESEARCH ERROR] request_id={request_id} error={type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the research query.",
        )
