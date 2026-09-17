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
from dotenv import load_dotenv

load_dotenv()

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
            pid = req.paper_id.strip()
            if pid.lower().startswith("paper_"):
                pid = pid[6:]
            if pid.upper().startswith("AGR") and len(pid) > 3 and pid[3:].isdigit():
                pid = "AG" + pid[3:]
            elif pid.upper().startswith("CS") and len(pid) > 2 and pid[2:].isdigit():
                pid = "AI" + pid[2:]
            filters["paper_id"] = pid
        elif req.uploaded_paper_name:
            clean_name = req.uploaded_paper_name.split("/")[-1].split("\\")[-1]
            filters["paper_id"] = clean_name.replace(".pdf", "").replace(".txt", "").replace(".md", "")

        logger.info(
            f"[SERVICE TRACE] request_id={request_id} user_id={user_id} question='{req.question}' "
            f"req.paper_id='{req.paper_id}' uploaded_paper_name='{req.uploaded_paper_name}' "
            f"has_uploaded_text={bool(req.uploaded_paper_text)} filters={filters}"
        )

        top_k = req.top_k or 10


        if req.paper_id:
            logger.info(f"[LOCAL-PAPER] request received in ResearchService | paper_id={req.paper_id} question='{req.question}'")

        try:
            # Execute core RAG Pipeline
            rag_res = self.pipeline.answer(
                question=req.question,
                filters=filters if filters else None,
                top_k=top_k,
                request_id=request_id,
                question_hash=question_hash,
                uploaded_paper_text=req.uploaded_paper_text,
                uploaded_paper_name=req.uploaded_paper_name,
            )

            # Convert RAGResponse to Pydantic ResearchQueryResponse schema
            res_dict = rag_res.to_dict()
            query_response = ResearchQueryResponse(**res_dict)

            # Log query history in database for authenticated user
            try:
                log_query_history(
                    db=db,
                    user_id=user_id,
                    rag_response=query_response,
                    domain=req.domain,
                    subtopic=req.subtopic,
                    paper_id=req.paper_id,
                )
            except Exception as hist_err:
                logger.warning(f"[SERVICE HISTORY WARNING] Failed to record query history: {hist_err}")

            return query_response
        except Exception as e:
            if req.paper_id:
                import traceback
                logger.error(f"[LOCAL-PAPER] EXCEPTION occurred during request processing for paper {req.paper_id}: {type(e).__name__}: {str(e)}")
                logger.error(f"[LOCAL-PAPER] TRACEBACK:\n{traceback.format_exc()}")
            raise

