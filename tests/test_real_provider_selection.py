"""
Phase 13/14 — Provider Selection & End-to-End LLM Answer Propagation Test Suite

Verifies:
1. Production ResearchService selects GeminiLLMProvider when GEMINI_API_KEY is present.
2. Missing GEMINI_API_KEY falls back cleanly to dynamic MockLLMProvider without crashing.
3. Raw Gemini LLM response flows 100% unchanged to POST /research/query API response.
"""

import os
import uuid
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base
from app.dependencies import get_db, reset_research_service
from app.services.research_service import ResearchService
from src.pipeline.llm import GeminiLLMProvider, MockLLMProvider

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
    reset_research_service()
    yield
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()
    reset_research_service()


class TestProviderSelectionAndPropagation:
    def test_gemini_provider_selected_when_key_present(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_test_key_12345"}, clear=False):
            service = ResearchService()
            assert isinstance(service.pipeline.llm, GeminiLLMProvider)
            assert service.pipeline.llm.api_key == "dummy_test_key_12345"

    def test_clean_fallback_when_key_absent(self):
        env_without_keys = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "LLM_API_KEY", "LLM_PROVIDER")}
        with patch.dict(os.environ, env_without_keys, clear=True):
            service = ResearchService()
            assert isinstance(service.pipeline.llm, MockLLMProvider)

    def test_gemini_answer_reaches_api_response(self):
        uid = str(uuid.uuid4())[:8]
        email = f"prop_test_{uid}@example.com"
        client.post("/auth/register", json={"email": email, "password": "Password123!"})
        login_res = client.post("/auth/login", data={"username": email, "password": "Password123!"})
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        expected_gemini_answer = "TEST_GEMINI_RESPONSE_FOR_AGRICULTURE [E1]"

        # Mock GeminiLLMProvider.generate to return custom answer
        with patch.object(GeminiLLMProvider, "generate", return_value=expected_gemini_answer):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy_test_key_12345"}, clear=False):
                reset_research_service()
                res = client.post(
                    "/research/query",
                    json={
                        "question": "How can IoT improve smart irrigation?",
                        "domain": "agriculture",
                        "top_k": 5
                    },
                    headers=headers
                )

                assert res.status_code == 200
                data = res.json()
                assert data["answer"] == expected_gemini_answer
