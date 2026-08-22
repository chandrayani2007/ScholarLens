"""
SQLAlchemy Database Models for Application Data (Users, Profiles, Saved Queries & History)

Rules:
- Stores ONLY user authentication data, profile details, saved queries, and user-isolated history records.
- Does NOT copy research corpus documents, chunks, vectors, or BM25 text into SQL.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from app.db.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    institution = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)
    academic_year = Column(String(100), nullable=True)
    research_interests = Column(Text, nullable=True)
    reset_token = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    query_histories = relationship("QueryHistory", back_populates="user", cascade="all, delete-orphan")
    saved_queries = relationship("SavedQuery", back_populates="user", cascade="all, delete-orphan")


class QueryHistory(Base):
    __tablename__ = "query_histories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    domain = Column(String(100), nullable=True)
    subtopic = Column(String(100), nullable=True)
    paper_id = Column(String(100), nullable=True)
    citations_json = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=True)
    why_this_answer_json = Column(Text, nullable=False)
    confidence = Column(String(50), nullable=False)
    limitations = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="query_histories")


class SavedQuery(Base):
    __tablename__ = "saved_queries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    domain = Column(String(100), nullable=True)
    citations_json = Column(Text, nullable=False)
    evidence_json = Column(Text, nullable=True)
    why_this_answer_json = Column(Text, nullable=False)
    confidence = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="saved_queries")
