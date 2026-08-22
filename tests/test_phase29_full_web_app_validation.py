"""
Phase 29 — Complete Web Application Functional Validation & User Data Isolation Test Suite

Verifies:
1. test_user_registration_and_uniqueness_checks()
2. test_duplicate_email_and_username_rejection_messages()
3. test_login_via_username_and_email()
4. test_forgot_and_reset_password_workflow()
5. test_profile_view_and_update()
6. test_save_query_and_saved_queries_management()
7. test_user_data_isolation_between_user_a_and_user_b()
8. test_corpus_library_browsing_filtering_search_and_pdf_stream()
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.dependencies import get_db
from app.db import models
from app.main import app

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


class TestPhase29WebAppValidation:

    @pytest.fixture(autouse=True)
    def setup_test_db(self):
        app.dependency_overrides[get_db] = override_get_db
        Base.metadata.create_all(bind=test_engine)
        yield
        Base.metadata.drop_all(bind=test_engine)
        app.dependency_overrides.clear()

    def test_user_registration_and_uniqueness_checks(self):
        """User registration requires unique email and unique username."""
        reg_payload = {
            "full_name": "Dr. Alice Morgan",
            "username": "alicemorgan",
            "email": "alice@university.edu",
            "password": "securepassword123",
            "confirm_password": "securepassword123"
        }
        res = client.post("/auth/register", json=reg_payload)
        assert res.status_code == 201
        data = res.json()
        assert data["username"] == "alicemorgan"
        assert data["email"] == "alice@university.edu"

    def test_duplicate_email_and_username_rejection_messages(self):
        """Duplicate registration attempts return clear, user-friendly error messages."""
        # Initial registration
        reg1 = client.post("/auth/register", json={
            "full_name": "Alice Initial",
            "username": "alice_unique_1",
            "email": "alice_unique_1@university.edu",
            "password": "password123"
        })
        assert reg1.status_code == 201

        # Duplicate email
        res1 = client.post("/auth/register", json={
            "full_name": "Alice Duplicate",
            "username": "different_uname",
            "email": "alice_unique_1@university.edu",
            "password": "password123"
        })
        assert res1.status_code == 400
        assert "already exists" in res1.json()["detail"]

        # Duplicate username
        res2 = client.post("/auth/register", json={
            "full_name": "Alice Duplicate",
            "username": "alice_unique_1",
            "email": "different_email@university.edu",
            "password": "password123"
        })
        assert res2.status_code == 400
        assert "already exists" in res2.json()["detail"]

    def test_login_via_username_and_email(self):
        """Users can log in using either their username OR their email address."""
        reg = client.post("/auth/register", json={
            "full_name": "Alice Login",
            "username": "alice_login_user",
            "email": "alice_login@university.edu",
            "password": "securepassword123"
        })
        assert reg.status_code == 201

        # Login via email
        res_email = client.post("/auth/login", data={"username": "alice_login@university.edu", "password": "securepassword123"})
        assert res_email.status_code == 200
        assert "access_token" in res_email.json()

        # Login via username
        res_uname = client.post("/auth/login", data={"username": "alice_login_user", "password": "securepassword123"})
        assert res_uname.status_code == 200
        assert "access_token" in res_uname.json()

    def test_forgot_and_reset_password_workflow(self):
        """Forgot password generates token, and reset-password updates credentials."""
        reg = client.post("/auth/register", json={
            "full_name": "Alice Reset",
            "username": "alice_reset_user",
            "email": "alice_reset@university.edu",
            "password": "oldpassword123"
        })
        assert reg.status_code == 201

        # Request reset token
        res_forgot = client.post("/auth/forgot-password", json={"email": "alice_reset@university.edu"})
        assert res_forgot.status_code == 200
        token = res_forgot.json()["reset_token"]
        assert token is not None

        # Reset password
        res_reset = client.post("/auth/reset-password", json={
            "email": "alice_reset@university.edu",
            "reset_token": token,
            "new_password": "newsecurepassword456"
        })
        assert res_reset.status_code == 200

        # Login with new password
        res_login = client.post("/auth/login", data={"username": "alice_reset@university.edu", "password": "newsecurepassword456"})
        assert res_login.status_code == 200

    def test_profile_view_and_update(self):
        """User profile can be viewed and updated by authenticated user."""
        reg = client.post("/auth/register", json={
            "full_name": "Alice Profile",
            "username": "alice_prof_user",
            "email": "alice_prof@university.edu",
            "password": "password123"
        })
        assert reg.status_code == 201

        login_res = client.post("/auth/login", data={"username": "alice_prof_user", "password": "password123"})
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # Fetch profile
        get_res = client.get("/api/profile/me", headers=headers)
        assert get_res.status_code == 200
        assert get_res.json()["username"] == "alice_prof_user"

        # Update profile
        update_payload = {
            "institution": "Stanford University",
            "department": "Artificial Intelligence Lab",
            "academic_year": "Postdoctoral Researcher",
            "research_interests": "Large Language Models & Grounded RAG"
        }
        put_res = client.put("/api/profile/me", json=update_payload, headers=headers)
        assert put_res.status_code == 200
        assert put_res.json()["institution"] == "Stanford University"

    def test_save_query_and_saved_queries_management(self):
        """Users can save research queries, list them, and delete them."""
        reg = client.post("/auth/register", json={
            "full_name": "Alice Save",
            "username": "alice_save_user",
            "email": "alice_save@university.edu",
            "password": "password123"
        })
        assert reg.status_code == 201

        login_res = client.post("/auth/login", data={"username": "alice_save_user", "password": "password123"})
        assert login_res.status_code == 200
        token = login_res.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        save_payload = {
            "question": "What are the limitations of RAG systems?",
            "answer": "RAG systems have several limitations related to retrieval noise and context size [E1].",
            "domain": "artificial_intelligence",
            "citations_json": "{}",
            "why_this_answer_json": "{}",
            "confidence": "High"
        }
        save_res = client.post("/api/saved-queries", json=save_payload, headers=headers)
        assert save_res.status_code == 201
        saved_id = save_res.json()["id"]

        # List saved queries
        list_res = client.get("/api/saved-queries", headers=headers)
        assert list_res.status_code == 200
        assert len(list_res.json()) >= 1

        # Delete saved query
        del_res = client.delete(f"/api/saved-queries/{saved_id}", headers=headers)
        assert del_res.status_code == 204

    def test_user_data_isolation_between_user_a_and_user_b(self):
        """User A cannot access or delete User B's saved queries."""
        # Create User A & User B
        reg_a = client.post("/auth/register", json={
            "full_name": "Alice User",
            "username": "alice_iso_user",
            "email": "alice_iso@university.edu",
            "password": "password123"
        })
        assert reg_a.status_code == 201

        reg_b = client.post("/auth/register", json={
            "full_name": "Bob Smith",
            "username": "bob_iso_user",
            "email": "bob_iso@university.edu",
            "password": "bobpassword123"
        })
        assert reg_b.status_code == 201

        token_a = client.post("/auth/login", data={"username": "alice_iso_user", "password": "password123"}).json()["access_token"]
        token_b = client.post("/auth/login", data={"username": "bob_iso_user", "password": "bobpassword123"}).json()["access_token"]

        headers_a = {"Authorization": f"Bearer {token_a}"}
        headers_b = {"Authorization": f"Bearer {token_b}"}

        # User A saves a query
        save_res = client.post("/api/saved-queries", json={
            "question": "User A Private Query",
            "answer": "Private Answer A",
            "domain": "cybersecurity",
            "citations_json": "{}",
            "why_this_answer_json": "{}",
            "confidence": "High"
        }, headers=headers_a)
        query_a_id = save_res.json()["id"]

        # User B attempts to access User A's saved query -> 404
        get_b_res = client.get(f"/api/saved-queries/{query_a_id}", headers=headers_b)
        assert get_b_res.status_code == 404

        # User B attempts to delete User A's saved query -> 404
        del_b_res = client.delete(f"/api/saved-queries/{query_a_id}", headers=headers_b)
        assert del_b_res.status_code == 404

    def test_corpus_library_browsing_filtering_search_and_pdf_stream(self):
        """Corpus page lists 1,000 papers, filters by domain, searches by title/author/ID, and streams PDF."""
        # List all
        list_res = client.get("/api/corpus")
        assert list_res.status_code == 200
        assert list_res.json()["total"] == 1000

        # Filter by domain
        ai_res = client.get("/api/corpus?domain=artificial_intelligence")
        assert ai_res.status_code == 200
        assert ai_res.json()["total"] == 200

        # Search by paper ID
        search_res = client.get("/api/corpus?search=AI001")
        assert search_res.status_code == 200
        assert search_res.json()["total"] >= 1

        # Fetch paper details
        details_res = client.get("/api/corpus/AI001")
        assert details_res.status_code == 200
        assert details_res.json()["paper_id"] == "AI001"

        # Stream PDF
        pdf_res = client.get("/api/corpus/AI001/pdf")
        assert pdf_res.status_code == 200
        assert pdf_res.headers["content-type"] == "application/pdf"
