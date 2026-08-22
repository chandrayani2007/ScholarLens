"""
Authentication Service (Bcrypt Hashing, JWT Tokens, User Profiles & Password Resets)

Rules:
- Uses direct bcrypt for secure password hashing.
- Uses JWT access tokens via jose.
- Enforces strict uniqueness for email and username.
- Supports password reset workflows safely without leaking sensitive information.
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.db.models import User
from app.schemas.auth import UserRegister, ProfileUpdate

JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev_secret_key_change_in_prod_123456789")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def register_user(db: Session, user_data: UserRegister) -> User:
    """Register a new application user with unique email and username verification."""
    if user_data.confirm_password and user_data.password != user_data.confirm_password:
        raise ValueError("Password confirmation does not match.")

    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise ValueError("An account with this email already exists. Please log in or use Forgot Password.")

    if not user_data.username:
        base_uname = user_data.email.split("@")[0].lower()
        clean_uname = "".join(c for c in base_uname if c.isalnum() or c in "_-") or "user"
        username = clean_uname
        counter = 1
        while db.query(User).filter(User.username == username).first():
            username = f"{clean_uname}_{counter}"
            counter += 1
    else:
        username = user_data.username.lower()
        existing_uname = db.query(User).filter(User.username == username).first()
        if existing_uname:
            raise ValueError("An account with this username already exists. Please choose another username.")

    hashed_pwd = hash_password(user_data.password)
    user = User(
        username=username,
        email=user_data.email,
        hashed_password=hashed_pwd,
        full_name=user_data.full_name or username.title(),
        institution="Research University",
        department="Computer Science & Engineering",
        academic_year="4th Year B.Tech",
        research_interests="Artificial Intelligence, Natural Language Processing, RAG",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


from typing import Optional, Tuple


def authenticate_user(db: Session, username_or_email: str, password: str) -> Tuple[Optional[User], Optional[str]]:
    """
    Authenticate user credentials using email OR username.
    Returns (user, None) on success, or (None, specific_error_message) on failure.
    """
    if not username_or_email or not str(username_or_email).strip():
        return None, "Please enter your username or email address."
    if not password:
        return None, "Please enter your password."

    identifier = str(username_or_email).strip().lower()
    user = db.query(User).filter((User.email == identifier) | (User.username == identifier)).first()
    if not user:
        if "@" in identifier:
            return None, "No account found with this email address. Please check the email or create an account."
        else:
            return None, "No account found with this username. Please check your username or create an account."

    if not verify_password(password, user.hashed_password):
        return None, "Incorrect password. Please try again or use Forgot Password to reset it."

    return user, None


def update_user_profile(db: Session, user: User, profile_data: ProfileUpdate) -> User:
    """Update profile information for the authenticated user."""
    if profile_data.email and profile_data.email != user.email:
        existing = db.query(User).filter(User.email == profile_data.email, User.id != user.id).first()
        if existing:
            raise ValueError("An account with this email address already exists.")
        user.email = profile_data.email

    if profile_data.username and profile_data.username != user.username:
        existing = db.query(User).filter(User.username == profile_data.username, User.id != user.id).first()
        if existing:
            raise ValueError("An account with this username already exists.")
        user.username = profile_data.username

    if profile_data.full_name is not None:
        user.full_name = profile_data.full_name
    if profile_data.institution is not None:
        user.institution = profile_data.institution
    if profile_data.department is not None:
        user.department = profile_data.department
    if profile_data.academic_year is not None:
        user.academic_year = profile_data.academic_year
    if profile_data.research_interests is not None:
        user.research_interests = profile_data.research_interests

    db.commit()
    db.refresh(user)
    return user


def generate_password_reset_token(db: Session, email: str) -> str:
    """Generate a secure password reset token for a valid account email."""
    email_clean = email.strip().lower()
    user = db.query(User).filter(User.email == email_clean).first()
    if not user:
        raise ValueError("No account found with this email address. Please check the email or sign up.")

    token = secrets.token_hex(16)
    user.reset_token = token
    db.commit()
    return token


def reset_user_password(db: Session, email: str, reset_token: str, new_password: str) -> bool:
    """Reset user password using valid token."""
    email_clean = email.strip().lower()
    user = db.query(User).filter(User.email == email_clean, User.reset_token == reset_token).first()
    if not user:
        raise ValueError("Invalid or expired password reset token.")

    user.hashed_password = hash_password(new_password)
    user.reset_token = None
    db.commit()
    return True
