"""
Health Monitoring Router

Endpoints:
- GET /health
- GET /health/retrieval

Safeguard:
Does NOT expose sensitive internal file paths, passwords, or secret keys.
"""

from fastapi import APIRouter, status, HTTPException
from app.services.research_service import ResearchService

router = APIRouter(tags=["Health Monitoring"])


@router.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Basic service liveness check."""
    return {"status": "ok", "service": "Research Mind Backend API"}


@router.get("/health/retrieval", status_code=status.HTTP_200_OK)
def retrieval_health_check():
    """Verify that ChromaDB and BM25 indexes are initialized and accessible."""
    try:
        service = ResearchService()
        retriever = service.pipeline.retriever

        chroma_record_count = retriever.chroma_indexer.collection.count()
        bm25_doc_count = len(retriever.bm25_indexer.units)

        return {
            "status": "healthy",
            "chromadb": {
                "accessible": True,
                "record_count": chroma_record_count,
            },
            "bm25": {
                "accessible": True,
                "document_count": bm25_doc_count,
            },
            "section_filtering_active": retriever.config.enable_section_filtering,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval services are currently unavailable.",
        )
