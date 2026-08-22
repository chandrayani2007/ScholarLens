"""
Research Query Router

Endpoints:
- POST /research/query
"""

import uuid
import hashlib
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import User
from app.dependencies import get_db, get_current_user, get_research_service
from app.schemas.query import ResearchQueryRequest
from app.schemas.response import ResearchQueryResponse
from app.services.research_service import ResearchService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["Research API"])

ALLOWED_DOMAINS = {"artificial_intelligence", "cybersecurity", "agriculture", "healthcare", "climate"}


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
