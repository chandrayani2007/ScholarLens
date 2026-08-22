"""
Saved Queries Router (Save, List, Fetch Details, Delete Saved Research Queries)

Endpoints:
- POST /api/saved-queries
- GET /api/saved-queries
- GET /api/saved-queries/{saved_id}
- DELETE /api/saved-queries/{saved_id}
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import User, SavedQuery
from app.dependencies import get_db, get_current_user
from app.schemas.auth import SavedQueryCreate, SavedQueryOut

router = APIRouter(prefix="/saved-queries", tags=["Saved Queries"])


@router.post("", response_model=SavedQueryOut, status_code=status.HTTP_201_CREATED)
def save_query(query_data: SavedQueryCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Save a research query result to the logged-in user's account."""
    # Check for duplicate save by question and user
    existing = db.query(SavedQuery).filter(
        SavedQuery.user_id == current_user.id,
        SavedQuery.question == query_data.question.strip()
    ).first()

    if existing:
        return existing

    saved = SavedQuery(
        user_id=current_user.id,
        question=query_data.question.strip(),
        answer=query_data.answer.strip(),
        domain=query_data.domain,
        citations_json=query_data.citations_json,
        why_this_answer_json=query_data.why_this_answer_json,
        confidence=query_data.confidence
    )
    db.add(saved)
    db.commit()
    db.refresh(saved)
    return saved


@router.get("", response_model=List[SavedQueryOut])
def list_saved_queries(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Retrieve all saved research queries for the authenticated user."""
    queries = db.query(SavedQuery).filter(SavedQuery.user_id == current_user.id).order_by(SavedQuery.created_at.desc()).all()
    return queries


@router.get("/{saved_id}", response_model=SavedQueryOut)
def get_saved_query(saved_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Retrieve a single saved query by ID (user isolated)."""
    saved = db.query(SavedQuery).filter(SavedQuery.id == saved_id, SavedQuery.user_id == current_user.id).first()
    if not saved:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved query not found or access denied.")
    return saved


@router.delete("/{saved_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved_query(saved_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Delete a saved query by ID (user isolated)."""
    saved = db.query(SavedQuery).filter(SavedQuery.id == saved_id, SavedQuery.user_id == current_user.id).first()
    if not saved:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved query not found or access denied.")
    db.delete(saved)
    db.commit()
    return None
