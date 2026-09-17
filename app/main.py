"""
Research Mind — FastAPI Backend Application Entrypoint

Architecture:
Frontend (Web / API Client)
    ↓
FastAPI App Layer (app/main.py)
    ↓
Authentication & Authorization (app/routers/auth.py, JWT)
    ↓
Routers:
- Authentication & Profile (/auth, /profile)
- Research Query Generator (/api/research)
- Saved Queries (/api/saved-queries)
- Query History (/api/history)
- Academic Paper Corpus & PDF Viewer (/api/corpus)
"""

import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import init_db
from app.routers import auth, profile, saved_queries, corpus, research, history, health

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Initialize SQL database tables
init_db()

app = FastAPI(
    title="ScholarLens Research Mind API",
    description="Evidence-Grounded Scientific Literature Search and RAG Engine API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Application Routers
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(profile.router, prefix="/api")
app.include_router(saved_queries.router, prefix="/api")
app.include_router(corpus.router, prefix="/api")
app.include_router(research.router)
app.include_router(history.router)


from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format Pydantic 422 validation errors into clean, human-readable error messages."""
    error_messages = []
    for err in exc.errors():
        msg = err.get("msg", "")
        cleaned = msg.replace("Value error, ", "").replace("value is not a valid email address: ", "")
        if "String should have at least" in cleaned:
            loc = err.get("loc", [])
            field = str(loc[-1]) if loc else "Field"
            if "password" in field.lower():
                cleaned = "Password must be at least 6 characters long."
            elif "username" in field.lower():
                cleaned = "Username must be at least 3 characters long."
            else:
                cleaned = f"{field.replace('_', ' ').title()} is too short."
        error_messages.append(cleaned)

    friendly_msg = ", ".join(error_messages) if error_messages else "Invalid input data provided."
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": friendly_msg},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure HTTP exceptions return clean structured JSON response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler preventing raw stack traces from leaking."""
    logger.error(f"Unhandled Exception on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again later."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
