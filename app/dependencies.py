"""
FastAPI Dependencies (Database Sessions, JWT Authentication, Service Injections)
"""

from typing import Generator, Optional
import os
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.db.database import SessionLocal, init_db
from app.db.models import User
from app.services.auth_service import decode_access_token
from app.services.research_service import ResearchService
from src.pipeline.llm import GeminiLLMProvider, MockLLMProvider

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

init_db()

_research_service_instance: Optional[ResearchService] = None


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials or token expired",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    user_id: Optional[int] = payload.get("user_id")
    email: Optional[str] = payload.get("sub")

    if user_id is not None:
        user = db.query(User).filter(User.id == user_id).first()
    elif email is not None:
        user = db.query(User).filter((User.email == email) | (User.username == email)).first()
    else:
        raise credentials_exception

    if user is None or not user.is_active:
        raise credentials_exception

    return user


def reset_research_service():
    global _research_service_instance
    _research_service_instance = None


def get_research_service() -> ResearchService:
    global _research_service_instance
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")

    if _research_service_instance is None:
        _research_service_instance = ResearchService()
    elif api_key and api_key.strip() and not isinstance(_research_service_instance.pipeline.llm, GeminiLLMProvider):
        logger.info("Upgrading ResearchService instance to GeminiLLMProvider based on active GEMINI_API_KEY.")
        _research_service_instance = ResearchService()

    return _research_service_instance
