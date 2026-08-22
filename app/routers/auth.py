"""
Authentication Router (Registration, Login, Profile, Forgot Password & Password Reset)

Endpoints:
- POST /auth/register
- POST /auth/login
- GET /auth/me
- PUT /auth/me
- POST /auth/forgot-password
- POST /auth/reset-password
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.db.models import User
from app.dependencies import get_db, get_current_user
from app.schemas.auth import (
    UserRegister, TokenResponse, UserOut, ProfileUpdate,
    ForgotPasswordRequest, ResetPasswordRequest
)
from app.services.auth_service import (
    register_user, authenticate_user, create_access_token,
    update_user_profile, generate_password_reset_token, reset_user_password
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    """Register a new application user with username and email uniqueness validation."""
    try:
        user = register_user(db, user_data)
        return user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Authenticate user using email OR username and return JWT access token."""
    user, error_msg = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_msg or "Incorrect username/email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.email, "user_id": user.id, "username": user.username})
    return TokenResponse(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    """Get profile of current authenticated user."""
    return current_user


@router.put("/me", response_model=UserOut)
def update_me(profile_data: ProfileUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Update profile of current authenticated user."""
    try:
        updated_user = update_user_profile(db, current_user, profile_data)
        return updated_user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/forgot-password")
def forgot_password(request_data: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Request password reset for a registered email address."""
    try:
        token = generate_password_reset_token(db, request_data.email)
        return {
            "message": "Password reset token generated successfully. In development environment, use this reset token to update your password.",
            "email": request_data.email,
            "reset_token": token
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@router.post("/reset-password")
def reset_password(request_data: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Reset user password using reset token."""
    try:
        reset_user_password(db, request_data.email, request_data.reset_token, request_data.new_password)
        return {"message": "Password reset successfully. You can now log in with your new password."}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
