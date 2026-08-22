"""
Reset application SQLite database tables to match updated SQLAlchemy schema.
"""

from pathlib import Path
from app.db.database import Base, engine, init_db
from app.db import models

db_file = Path("data/app.db")
if db_file.exists():
    db_file.unlink()

init_db()
print("Database schema reset and initialized successfully.")
