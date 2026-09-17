import sys
import os

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath("."))

from app.db.database import SessionLocal
from app.db.models import User
from app.schemas.query import ResearchQueryRequest
from app.services.research_service import ResearchService

db = SessionLocal()
user = db.query(User).first()
research_service = ResearchService()

test_cases = [
    {"paper_id": "CS001", "q": "what is the main objective of this paper?"},
    {"paper_id": "CS001", "q": "what methodology does this paper propose?"},
    {"paper_id": "AGR001", "q": "what problem does it address?"},
    {"paper_id": "HC001", "q": "what are the quantitative results?"},
    {"paper_id": "CY001", "q": "what dataset is used?"},
    {"paper_id": "CL001", "q": "what are the limitations?"}
]

print("=== CORPUS-WIDE MULTI-DOMAIN GENERALIZATION AUDIT ===")

for tc in test_cases:
    pid = tc["paper_id"]
    q = tc["q"]
    req = ResearchQueryRequest(
        question=q,
        paper_id=pid,
        selected_corpus=[pid],
        enable_web_search=False
    )
    res = research_service.process_query(db, user.id, req)
    print(f"\n=======================================================")
    print(f"PAPER: {pid} | QUESTION: '{q}'")
    print(f"=======================================================")
    print("ANSWER:")
    print(res.answer)
    print("CONFIDENCE:", res.confidence)
    print("CITATIONS:", [c.citation_tag for c in res.citations if hasattr(c, 'citation_tag')] if hasattr(res, 'citations') else [])
