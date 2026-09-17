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
    {
        "paper_id": "AI001",
        "title": "Paper AI001",
        "questions": [
            "what is the main objective of this paper?",
            "what problem does it address?",
            "what methodology does this paper propose?"
        ]
    },
    {
        "paper_id": "AI002",
        "title": "Paper AI002",
        "questions": [
            "what is the main objective of this paper?",
            "what problem does it address?",
            "what methodology does this paper propose?"
        ]
    },
    {
        "paper_id": "AI005",
        "title": "Paper AI005",
        "questions": [
            "what is the main objective of this paper?",
            "what problem does it address?",
            "what dataset is used?"
        ]
    }
]

print("=== STARTING MULTI-PAPER GENERALIZATION AUDIT ===")

for case in test_cases:
    pid = case["paper_id"]
    print(f"\n=======================================================")
    print(f"AUDITING PAPER: {pid} ({case['title']})")
    print(f"=======================================================")
    for q in case["questions"]:
        req = ResearchQueryRequest(
            question=q,
            paper_id=pid,
            selected_corpus=[pid],
            enable_web_search=False
        )
        res = research_service.process_query(db, user.id, req)
        print(f"\n--- QUESTION: '{q}' ---")
        print("ANSWER:")
        print(res.answer)
        print("CONFIDENCE:", res.confidence)
        print("CITATIONS:", [c.citation_tag for c in res.citations if hasattr(c, 'citation_tag')] if hasattr(res, 'citations') else [])
