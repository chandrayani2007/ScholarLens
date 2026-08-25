"""
Query History Service (User-Isolated Application History Management)

Enforces strict user isolation:
Every query history read/write operation is scoped directly to user_id.
User A can NEVER access User B's query history.
"""

import json
from typing import List, Optional
from sqlalchemy.orm import Session

from app.db.models import QueryHistory
from app.schemas.response import ResearchQueryResponse


def log_query_history(
    db: Optional[Session],
    user_id: int,
    rag_response: ResearchQueryResponse,
    domain: Optional[str] = None,
    subtopic: Optional[str] = None,
    paper_id: Optional[str] = None,
) -> Optional[QueryHistory]:
    """Save user query response summary and full evidence into SQL application database."""
    if db is None:
        return None

    evidence_data = [
        ev.model_dump() if hasattr(ev, "model_dump") else ev.dict()
        for ev in rag_response.evidence
    ] if rag_response.evidence else []

    history = QueryHistory(
        user_id=user_id,
        question=rag_response.question,
        answer=rag_response.answer,
        domain=domain or (rag_response.retrieval_metadata.get("domain") if rag_response.retrieval_metadata else "general"),
        subtopic=subtopic,
        paper_id=paper_id,
        citations_json=json.dumps({k: v.model_dump() if hasattr(v, "model_dump") else v.dict() for k, v in rag_response.citations.items()}),
        evidence_json=json.dumps(evidence_data),
        why_this_answer_json=json.dumps(rag_response.why_this_answer.model_dump() if hasattr(rag_response.why_this_answer, "model_dump") else rag_response.why_this_answer.dict()),
        confidence=rag_response.confidence,
        limitations=rag_response.limitations,
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return history


def get_user_history(db: Session, user_id: int, limit: int = 50, offset: int = 0) -> List[QueryHistory]:
    """Retrieve list of query history records owned exclusively by user_id."""
    return (
        db.query(QueryHistory)
        .filter(QueryHistory.user_id == user_id)
        .order_by(QueryHistory.timestamp.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_user_history_detail(db: Session, user_id: int, query_id: int) -> Optional[QueryHistory]:
    """Retrieve single query history record verifying ownership by user_id."""
    return (
        db.query(QueryHistory)
        .filter(QueryHistory.id == query_id, QueryHistory.user_id == user_id)
        .first()
    )
