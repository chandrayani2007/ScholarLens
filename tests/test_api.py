"""
Phase 8 — FastAPI Integration Test Suite (Auth, Health, Research Queries, Query History & User Isolation)
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.dependencies import get_db
from app.db.database import Base
from app.db.models import User, QueryHistory


# Setup in-memory SQLite database for testing with StaticPool so all connections share the same memory DB
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_test_db():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def client():
    return TestClient(app)


def get_auth_headers(client, email="user@example.com", password="password123"):
    client.post("/auth/register", json={"email": email, "password": password})
    res = client.post("/auth/login", data={"username": email, "password": password})
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestHealthEndpoints:
    def test_health_check(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

    def test_retrieval_health_check(self, client):
        res = client.get("/health/retrieval")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["chromadb"]["accessible"] is True
        assert data["chromadb"]["record_count"] > 0
        assert data["bm25"]["accessible"] is True
        assert data["bm25"]["document_count"] > 0


class TestAuthentication:
    def test_registration_success(self, client):
        res = client.post("/auth/register", json={"email": "test@example.com", "password": "securepassword"})
        assert res.status_code == 201
        data = res.json()
        assert data["email"] == "test@example.com"
        assert "id" in data
        assert "password" not in data

    def test_duplicate_registration_fails(self, client):
        client.post("/auth/register", json={"email": "test@example.com", "password": "securepassword"})
        res = client.post("/auth/register", json={"email": "test@example.com", "password": "securepassword"})
        assert res.status_code == 400
        assert "already exists" in res.json()["detail"]

    def test_login_success(self, client):
        client.post("/auth/register", json={"email": "login@example.com", "password": "password123"})
        res = client.post("/auth/login", data={"username": "login@example.com", "password": "password123"})
        assert res.status_code == 200
        data = res.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_password_fails(self, client):
        client.post("/auth/register", json={"email": "login@example.com", "password": "password123"})
        res = client.post("/auth/login", data={"username": "login@example.com", "password": "wrongpassword"})
        assert res.status_code == 401

    def test_get_me_profile(self, client):
        headers = get_auth_headers(client, email="me@example.com")
        res = client.get("/auth/me", headers=headers)
        assert res.status_code == 200
        assert res.json()["email"] == "me@example.com"

    def test_unauthorized_access_fails(self, client):
        res = client.get("/auth/me")
        assert res.status_code == 401


class TestResearchQueryAPI:
    def test_protected_research_endpoint(self, client):
        res = client.post("/research/query", json={"question": "What is RAG?"})
        assert res.status_code == 401

    def test_research_query_success(self, client):
        headers = get_auth_headers(client)
        res = client.post(
            "/research/query",
            json={"question": "What is retrieval augmented generation?", "domain": "artificial_intelligence"},
            headers=headers,
        )
        assert res.status_code == 200
        data = res.json()
        assert data["question"] == "What is retrieval augmented generation?"
        assert "answer" in data
        assert "citations" in data
        assert "why_this_answer" in data
        assert data["confidence"] in {"Excellent", "High", "Medium", "Low", "Insufficient"}

    def test_invalid_domain_fails(self, client):
        headers = get_auth_headers(client)
        res = client.post(
            "/research/query",
            json={"question": "What is RAG?", "domain": "invalid_domain_xyz"},
            headers=headers,
        )
        assert res.status_code == 400
        assert "Invalid domain" in res.json()["detail"]


class TestQueryHistoryAndIsolation:
    def test_query_history_created_and_retrieved(self, client):
        headers = get_auth_headers(client, email="hist@example.com")
        client.post(
            "/research/query",
            json={"question": "What is RAG?", "domain": "artificial_intelligence"},
            headers=headers,
        )

        res = client.get("/history", headers=headers)
        assert res.status_code == 200
        history = res.json()
        assert len(history) == 1
        item = history[0]
        assert item["question"] == "What is RAG?"
        assert "confidence" in item
        assert item["domain"] == "artificial_intelligence"

    def test_user_history_isolation(self, client):
        headers1 = get_auth_headers(client, email="user1@example.com")
        client.post(
            "/research/query",
            json={"question": "User 1 Question", "domain": "artificial_intelligence"},
            headers=headers1,
        )

        headers2 = get_auth_headers(client, email="user2@example.com")
        client.post(
            "/research/query",
            json={"question": "User 2 Question", "domain": "cybersecurity"},
            headers=headers2,
        )

        res1 = client.get("/history", headers=headers1)
        res2 = client.get("/history", headers=headers2)

        assert len(res1.json()) == 1
        assert res1.json()[0]["question"] == "User 1 Question"

        assert len(res2.json()) == 1
        assert res2.json()[0]["question"] == "User 2 Question"
