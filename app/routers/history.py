"""
Query History Router

Endpoints:
- GET /history
- GET /history/{query_id}

User Isolation Safeguard:
Users can ONLY view their own query history records. User A cannot access User B's history.
"""

import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import User
from app.dependencies import get_db, get_current_user
from app.schemas.response import QueryHistorySummary, QueryHistoryDetail
from app.services.history_service import get_user_history, get_user_history_detail

router = APIRouter(prefix="/history", tags=["Query History"])


@router.get("", response_model=List[QueryHistorySummary])
def list_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve list of query history records owned exclusively by the authenticated user."""
    records = get_user_history(db=db, user_id=current_user.id, limit=limit, offset=offset)
    return [
        QueryHistorySummary(
            id=r.id,
            question=r.question,
            domain=r.domain,
            confidence=r.confidence,
            timestamp=r.timestamp,
            created_at=r.timestamp,
        )
        for r in records
    ]


@router.get("/{query_id}", response_model=QueryHistoryDetail)
def get_history_item(
    query_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieve detailed query history record owned by the authenticated user."""
    record = get_user_history_detail(db=db, user_id=current_user.id, query_id=query_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query history record {query_id} not found.",
        )

    evidence_list = []
    if record.evidence_json:
        try:
            evidence_list = json.loads(record.evidence_json)
        except Exception:
            evidence_list = []

    return QueryHistoryDetail(
        id=record.id,
        user_id=record.user_id,
        question=record.question,
        answer=record.answer,
        domain=record.domain,
        subtopic=record.subtopic,
        paper_id=record.paper_id,
        citations=json.loads(record.citations_json) if record.citations_json else {},
        evidence=evidence_list,
        why_this_answer=json.loads(record.why_this_answer_json) if record.why_this_answer_json else {},
        confidence=record.confidence,
        limitations=record.limitations,
        timestamp=record.timestamp,
        created_at=record.timestamp,
    )
