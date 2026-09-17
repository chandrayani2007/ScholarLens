import os
import sys
sys.path.insert(0, os.path.abspath("."))
from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from app.main import app
from app.db.database import SessionLocal
from app.db.models import User
from app.services.auth_service import create_access_token

db = SessionLocal()
user = db.query(User).first()
if not user:
    user = User(email="test@scholar.lens", hashed_password="pw", full_name="Test User")
    db.add(user)
    db.commit()
    db.refresh(user)

token = create_access_token(data={"sub": user.email, "user_id": user.id, "username": user.username})
client = TestClient(app)

print(f"Testing /research/query with token for user {user.email}...")
headers = {"Authorization": f"Bearer {token}"}

payload = {
    "question": "What research problem does this paper address?",
    "paper_id": "AI038",
    "domain": "artificial_intelligence",
    "top_k": 10
}

response = client.post("/research/query", json=payload, headers=headers)
print("Status Code:", response.status_code)
if response.status_code == 200:
    data = response.json()
    print("Response keys:", list(data.keys()))
    print("Answer preview:\n", data.get("answer")[:300])
    print("Evidence count:", len(data.get("evidence", [])))
    print("Citations count:", len(data.get("citations", {})))
    print("Why this answer:", data.get("why_this_answer"))
else:
    print("Error response:", response.text)
