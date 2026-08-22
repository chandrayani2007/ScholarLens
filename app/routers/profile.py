"""
Profile Router (View & Edit Profile)

Endpoints:
- GET /api/profile/me
- PUT /api/profile/me
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import User
from app.dependencies import get_db, get_current_user
from app.schemas.auth import UserOut, ProfileUpdate
from app.services.auth_service import update_user_profile

router = APIRouter(prefix="/profile", tags=["Profile"])


@router.get("/me", response_model=UserOut)
def get_user_profile(current_user: User = Depends(get_current_user)):
    """Retrieve profile of currently authenticated user."""
    return current_user


@router.put("/me", response_model=UserOut)
def update_profile(profile_data: ProfileUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update profile information of currently authenticated user."""
    try:
        user = update_user_profile(db, current_user, profile_data)
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
