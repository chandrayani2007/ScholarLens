"""
Database Initialization, Auto-Migration and SQLAlchemy Session Setup

Application data (users, profiles, saved queries, query history) is stored in SQL (SQLite for dev/testing, PostgreSQL for production).
The research corpus (PDFs, ChromaDB, BM25, chunks) remains uncopied in its existing vector/file indexes.
"""

import os
import logging
from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_URL = f"sqlite:///{BASE_DIR / 'data' / 'app.db'}"

DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DB_URL)

# SQLite-specific connect args
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def auto_migrate_sqlite():
    """Auto-migrate SQLite database columns for existing local tables."""
    if not DATABASE_URL.startswith("sqlite"):
        return

    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        with engine.connect() as conn:
            # 1. Check users table columns
            if "users" in tables:
                cols = {c["name"] for c in inspector.get_columns("users")}
                migrations = {
                    "username": "VARCHAR(100)",
                    "full_name": "VARCHAR(255)",
                    "institution": "VARCHAR(255)",
                    "department": "VARCHAR(255)",
                    "academic_year": "VARCHAR(100)",
                    "research_interests": "TEXT",
                    "reset_token": "VARCHAR(255)",
                }
                for col_name, col_type in migrations.items():
                    if col_name not in cols:
                        logger.info(f"[DB AUTO-MIGRATE] Adding column users.{col_name}...")
                        conn.execute(text(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}"))

            # 2. Check query_histories table columns
            if "query_histories" in tables:
                cols = {c["name"] for c in inspector.get_columns("query_histories")}
                if "evidence_json" not in cols:
                    logger.info("[DB AUTO-MIGRATE] Adding column query_histories.evidence_json...")
                    conn.execute(text("ALTER TABLE query_histories ADD COLUMN evidence_json TEXT"))

            # 3. Check saved_queries table columns
            if "saved_queries" in tables:
                cols = {c["name"] for c in inspector.get_columns("saved_queries")}
                if "evidence_json" not in cols:
                    logger.info("[DB AUTO-MIGRATE] Adding column saved_queries.evidence_json...")
                    conn.execute(text("ALTER TABLE saved_queries ADD COLUMN evidence_json TEXT"))

            conn.commit()
    except Exception as e:
        logger.warning(f"[DB AUTO-MIGRATE WARNING] {type(e).__name__}: {e}")


def init_db():
    """Create all application database tables and auto-migrate missing columns."""
    from app.db import models  # Ensure models are imported for metadata registration
    from app.services.auth_service import seed_default_user
    Base.metadata.create_all(bind=engine)
    auto_migrate_sqlite()
    with SessionLocal() as db:
        seed_default_user(db)

