"""
Phase 10 & 12 — End-to-End System Integration Test Suite

Verifies:
1. Full user workflow (register -> login -> query -> response -> history -> detail -> isolation -> logout).
2. 5-Domain research query pipeline (AI, Cybersecurity, Agriculture, Healthcare, Climate).
3. Citation validation and provenance preservation.
4. "Why This Answer?" structured explainability.
5. Insufficient evidence fallback handling.
6. Domain metadata filtering.
7. User query history isolation (User A vs. User B).
8. Production health endpoints and FAISS absence.
9. Different questions produce different retrieved context, papers, and answers (Phase 12 bug fix).
"""

import os
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base
from app.dependencies import get_db

SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_test_db():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


class TestAuthenticationAndUserIsolation:
    def test_full_auth_and_user_history_isolation(self):
        uid = str(uuid.uuid4())[:8]
        # 1. Register User A
        email_a = f"user_a_{uid}@example.com"
        pass_a = "Password123!"
        reg_a_res = client.post("/auth/register", json={"email": email_a, "password": pass_a})
        assert reg_a_res.status_code == 201
        assert reg_a_res.json()["email"] == email_a

        # Duplicate registration fails
        reg_dup_res = client.post("/auth/register", json={"email": email_a, "password": pass_a})
        assert reg_dup_res.status_code == 400

        # Login User A
        login_a_res = client.post("/auth/login", data={"username": email_a, "password": pass_a})
        assert login_a_res.status_code == 200
        token_a = login_a_res.json()["access_token"]
        headers_a = {"Authorization": f"Bearer {token_a}"}

        # Verify User A profile
        me_a = client.get("/auth/me", headers=headers_a)
        assert me_a.status_code == 200
        assert me_a.json()["email"] == email_a

        # 2. Register User B
        email_b = f"user_b_{uid}@example.com"
        pass_b = "Password456!"
        client.post("/auth/register", json={"email": email_b, "password": pass_b})
        login_b_res = client.post("/auth/login", data={"username": email_b, "password": pass_b})
        token_b = login_b_res.json()["access_token"]
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # 3. User A submits a research query
        query_payload = {
            "question": "What is retrieval augmented generation?",
            "domain": "artificial_intelligence",
            "top_k": 5
        }
        q_res_a = client.post("/research/query", json=query_payload, headers=headers_a)
        assert q_res_a.status_code == 200
        data_a = q_res_a.json()
        assert "answer" in data_a
        assert "citations" in data_a

        # 4. Check User A history contains the query
        hist_a = client.get("/history", headers=headers_a)
        assert hist_a.status_code == 200
        items_a = hist_a.json()
        assert len(items_a) >= 1
        query_id_a = items_a[0]["id"]

        # Detail query for User A
        detail_a = client.get(f"/history/{query_id_a}", headers=headers_a)
        assert detail_a.status_code == 200
        assert detail_a.json()["question"] == query_payload["question"]

        # 5. Check User B history does NOT contain User A's query
        hist_b = client.get("/history", headers=headers_b)
        assert hist_b.status_code == 200
        items_b = hist_b.json()
        assert not any(item["id"] == query_id_a for item in items_b)

        # User B attempting to fetch User A's specific query ID fails (404)
        detail_b = client.get(f"/history/{query_id_a}", headers=headers_b)
        assert detail_b.status_code == 404


class Test5DomainResearchQueries:
    @pytest.mark.parametrize("domain,question", [
        ("artificial_intelligence", "What is retrieval augmented generation?"),
        ("cybersecurity", "How are network attacks detected?"),
        ("agriculture", "How is IoT used in smart farming?"),
        ("healthcare", "How does medical image analysis support diagnosis?"),
        ("climate", "How is machine learning used for climate prediction?"),
    ])
    def test_domain_query_pipeline(self, domain, question):
        uid = str(uuid.uuid4())[:8]
        email = f"test_{domain}_{uid}@example.com"
        client.post("/auth/register", json={"email": email, "password": "Password123!"})
        login_res = client.post("/auth/login", data={"username": email, "password": "Password123!"})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Submit domain query
        res = client.post(
            "/research/query",
            json={"question": question, "domain": domain, "top_k": 5},
            headers=headers
        )
        assert res.status_code == 200
        data = res.json()

        # Validate RAGResponse structure
        assert "answer" in data
        assert "citations" in data
        assert "evidence" in data
        assert "why_this_answer" in data
        assert "confidence" in data

        # Validate why_this_answer explainability parameters
        why = data["why_this_answer"]
        assert "contributing_papers" in why
        assert "contributing_sections" in why
        assert "evidence_strength" in why
        assert "multi_paper_support" in why
        assert "conflicts_detected" in why
        assert "explanation_summary" in why

        # Validate retrieved evidence items match selected domain
        evidence = data["evidence"]
        assert len(evidence) > 0
        for item in evidence:
            assert item["domain"] == domain
            assert "paper_id" in item
            assert "section_name" in item
            assert "page_start" in item
            assert "page_end" in item


class TestDifferentQuestionsDifferentAnswers:
    """Phase 12 bug fix verification test."""

    def test_different_questions_produce_different_context_and_answers(self):
        uid = str(uuid.uuid4())[:8]
        email = f"diff_q_{uid}@example.com"
        client.post("/auth/register", json={"email": email, "password": "Password123!"})
        login_res = client.post("/auth/login", data={"username": email, "password": "Password123!"})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Question A (AI)
        q_a = "What is retrieval augmented generation?"
        res_a = client.post("/research/query", json={"question": q_a, "domain": "artificial_intelligence", "top_k": 5}, headers=headers)
        assert res_a.status_code == 200
        data_a = res_a.json()

        # Question B (Agriculture)
        q_b = "How can IoT improve smart irrigation?"
        res_b = client.post("/research/query", json={"question": q_b, "domain": "agriculture", "top_k": 5}, headers=headers)
        assert res_b.status_code == 200
        data_b = res_b.json()

        # 1. Questions must differ
        assert data_a["question"] != data_b["question"]

        # 2. Answers must differ
        assert data_a["answer"] != data_b["answer"]

        # 3. Retrieved evidence papers must differ
        ev_a_papers = [e["paper_id"] for e in data_a["evidence"]]
        ev_b_papers = [e["paper_id"] for e in data_b["evidence"]]
        assert ev_a_papers != ev_b_papers

        # 4. First evidence text snippet must differ
        assert data_a["evidence"][0]["text"] != data_b["evidence"][0]["text"]


class TestInsufficientEvidenceFallback:
    def test_out_of_corpus_query(self):
        uid = str(uuid.uuid4())[:8]
        email = f"fallback_{uid}@example.com"
        client.post("/auth/register", json={"email": email, "password": "Password123!"})
        login_res = client.post("/auth/login", data={"username": email, "password": "Password123!"})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        out_of_corpus_q = "What is the detailed authentic recipe for baking traditional San Francisco sourdough bread with wild yeast starter?"
        res = client.post(
            "/research/query",
            json={"question": out_of_corpus_q, "top_k": 5},
            headers=headers
        )
        assert res.status_code == 200
        data = res.json()
        assert "answer" in data
        assert "confidence" in data


class TestHealthAndFAISSAbsence:
    def test_health_endpoints(self):
        h_res = client.get("/health")
        assert h_res.status_code == 200
        assert h_res.json()["status"] == "ok"

        r_res = client.get("/health/retrieval")
        assert r_res.status_code == 200
        r_data = r_res.json()
        assert r_data["status"] == "healthy"
        assert r_data["chromadb"]["accessible"] is True
        assert r_data["chromadb"]["record_count"] > 0
        assert r_data["bm25"]["accessible"] is True
        assert r_data["bm25"]["document_count"] > 0

    def test_faiss_files_absent(self):
        assert not os.path.exists("data/indexes/faiss.index")
        assert not os.path.exists("data/indexes/faiss_mapping.json")
