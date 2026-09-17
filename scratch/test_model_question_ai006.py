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

q = "what models are actually used?"
pid = "AI006"

req = ResearchQueryRequest(
    question=q,
    paper_id=pid,
    selected_corpus=[pid],
    enable_web_search=False
)

res = research_service.process_query(db, user.id, req)

print("==========================================")
print(f"PAPER: {pid} | QUESTION: '{q}'")
print("==========================================")
print("ANSWER:")
print(res.answer)
print("CONFIDENCE:", res.confidence)
print("CITATIONS:", [c.citation_tag for c in res.citations if hasattr(c, 'citation_tag')] if hasattr(res, 'citations') else [])
