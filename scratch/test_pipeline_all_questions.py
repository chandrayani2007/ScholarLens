import sys
import os

sys.path.insert(0, os.path.abspath("."))

from app.db.database import SessionLocal
from app.db.models import User
from app.schemas.query import ResearchQueryRequest
from app.services.research_service import ResearchService

db = SessionLocal()
user = db.query(User).first()
research_service = ResearchService()

questions = [
    "what is the main objective of this paper?",
    "what problem does it address?",
    "what methodology does this paper propose?",
    "what are the quantitative results?",
    "what dataset is used?",
    "what are the limitations?",
    "what is the future work?"
]

print("=== AUDITING PAPER AI003 (Deepchecks) ===")
for q in questions:
    req = ResearchQueryRequest(
        question=q,
        paper_id="AI003",
        selected_corpus=["AI003"],
        enable_web_search=False
    )
    res = research_service.process_query(db, user.id, req)
    print("==========================================")
    print(f"QUESTION: {q}")
    print("ANSWER:")
    print(res.answer)
    print("CITATIONS:", [c.citation_tag for c in res.citations if hasattr(c, 'citation_tag')] if hasattr(res, 'citations') else [])
