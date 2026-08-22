"""
Pydantic Schemas for Research Query Responses and History Outputs
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict


class CitationSchema(BaseModel):
    citation_id: str
    paper_id: str
    section_name: str
    pages: str
    chunk_id: str
    unit_id: str
    source_type: Optional[str] = "corpus"
    title: Optional[str] = None
    authors: List[str] = []
    published: Optional[str] = None
    url: Optional[str] = None


class EvidenceItemSchema(BaseModel):
    citation_id: str
    unit_id: str
    parent_chunk_id: str
    chunk_id: str
    paper_id: str
    section_id: str
    section_name: str
    domain: str
    subtopic: str
    page_start: int
    page_end: int
    text: str
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rrf_score: float = 0.0
    retrieval_methods: List[str]
    source_type: Optional[str] = "corpus"
    url: Optional[str] = None
    title: Optional[str] = None
    authors: List[str] = []
    published_date: Optional[str] = None


class WhyThisAnswerSchema(BaseModel):
    contributing_papers: List[str]
    contributing_sections: List[str]
    evidence_passages: List[str]
    multi_paper_support: bool
    evidence_strength: str
    inference_present: bool
    conflicts_detected: bool
    explanation_summary: str
    bullet_points: List[str] = []
    source_type: Optional[str] = "Research Mind Corpus"
    domain_scope: Optional[str] = "All Domains"


class ResearchQueryResponse(BaseModel):
    question: str
    answer: str
    citations: Dict[str, CitationSchema]
    evidence: List[EvidenceItemSchema]
    why_this_answer: WhyThisAnswerSchema
    confidence: str
    confidence_rationale: str
    limitations: str
    retrieval_metadata: Dict[str, Any]
    domain_scope: Optional[List[str]] = None
    request_id: Optional[str] = None
    question_hash: Optional[str] = None


class QueryHistorySummary(BaseModel):
    id: int
    question: str
    domain: Optional[str] = None
    confidence: str
    timestamp: Optional[datetime] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class QueryHistoryDetail(BaseModel):
    id: int
    question: str
    answer: str
    domain: Optional[str] = None
    subtopic: Optional[str] = None
    paper_id: Optional[str] = None
    citations: Dict[str, Any]
    evidence: List[Dict[str, Any]] = []
    why_this_answer: Dict[str, Any]
    confidence: str
    limitations: Optional[str] = None
    timestamp: Optional[datetime] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
