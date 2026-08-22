"""
Research Service (Pipeline Orchestration & API Bridge)

Rules:
- Bridges FastAPI requests directly to the existing src/pipeline RAGPipeline & HybridRetriever.
- Does NOT duplicate retrieval, RRF, section filtering, or RAG logic inside routers or services.
"""

import os
import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from src.pipeline.rag import RAGPipeline
from src.pipeline.retrieval import HybridRetriever
from src.pipeline.llm import get_llm_provider
from app.schemas.query import ResearchQueryRequest
from app.schemas.response import ResearchQueryResponse
from app.services.history_service import log_query_history

logger = logging.getLogger(__name__)


class ResearchService:
    """Service wrapping the core RAGPipeline for FastAPI request processing."""

    def __init__(self, rag_pipeline: Optional[RAGPipeline] = None):
        if rag_pipeline is None:
            # Check environment for API keys and test mode
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
            provider_env = os.environ.get("LLM_PROVIDER", "").lower()

            if (provider_env == "gemini" or api_key) and api_key and api_key.strip():
                logger.info("[ResearchService] Initializing live GeminiLLMProvider (gemini-2.5-flash)...")
                llm_provider = get_llm_provider("gemini")
            else:
                logger.info("[ResearchService] GEMINI_API_KEY not configured. Using dynamic MockLLMProvider.")
                llm_provider = get_llm_provider("mock")

            retriever = HybridRetriever()
            self.pipeline = RAGPipeline(retriever=retriever, llm_provider=llm_provider)
        else:
            self.pipeline = rag_pipeline

    def process_query(
        self,
        db: Session,
        user_id: int,
        req: ResearchQueryRequest,
        request_id: Optional[str] = None,
        question_hash: Optional[str] = None,
    ) -> ResearchQueryResponse:
        """
        Execute RAG query via core pipeline, log query history, and return structured ResearchQueryResponse.
        """
        filters: Dict[str, Any] = {}
        if req.domain:
            filters["domain"] = req.domain
        if req.subtopic:
            filters["subtopic"] = req.subtopic
        if req.paper_id:
            filters["paper_id"] = req.paper_id

        top_k = req.top_k or 10

        # Execute core RAG Pipeline
        rag_res = self.pipeline.answer(
            question=req.question,
            filters=filters if filters else None,
            top_k=top_k,
            request_id=request_id,
            question_hash=question_hash,
        )

        # Convert RAGResponse to Pydantic ResearchQueryResponse schema
        res_dict = rag_res.to_dict()
        query_response = ResearchQueryResponse(**res_dict)

        # Log query history in database for authenticated user
        log_query_history(
            db=db,
            user_id=user_id,
            rag_response=query_response,
            domain=req.domain,
            subtopic=req.subtopic,
            paper_id=req.paper_id,
        )

        return query_response
