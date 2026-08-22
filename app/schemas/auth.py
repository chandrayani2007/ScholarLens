"""
Pydantic Schemas for Authentication, User Profile, Forgot Password & Saved Queries
"""

import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict


class UserRegister(BaseModel):
    full_name: Optional[str] = Field(None, description="User full name")
    username: Optional[str] = Field(None, description="Unique username")
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password must be at least 6 characters")
    confirm_password: Optional[str] = Field(None, description="Password confirmation")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not v or "@" not in v:
            raise ValueError("Please provide a valid email address (e.g. researcher@university.edu).")

        parts = v.split("@")
        if len(parts) != 2:
            raise ValueError("Email address must contain exactly one '@' symbol.")

        local_part, domain_part = parts[0], parts[1]

        if len(local_part) < 2 or len(local_part) > 64:
            raise ValueError("Email username must be at least 2 characters long.")

        if local_part.startswith(".") or local_part.endswith(".") or ".." in local_part:
            raise ValueError("Email username cannot start, end, or contain consecutive dots.")

        if not re.match(r"^[a-zA-Z0-9_.+-]+$", local_part):
            raise ValueError("Email username contains invalid characters.")

        if "." not in domain_part:
            raise ValueError("Email domain must include a valid top-level domain (e.g. .com, .edu, .org).")

        domain_labels = domain_part.split(".")
        if len(domain_labels) < 2:
            raise ValueError("Email domain must include a valid top-level domain.")

        for label in domain_labels:
            if len(label) < 1 or len(label) > 63:
                raise ValueError("Email domain format is invalid.")
            if not re.match(r"^[a-zA-Z0-9-]+$", label) or label.startswith("-") or label.endswith("-"):
                raise ValueError("Email domain contains invalid characters.")

        tld = domain_labels[-1]
        if len(tld) < 2 or not tld.isalpha():
            raise ValueError("Email must end with a valid top-level domain (e.g. .com, .edu, .org, .net).")

        return v

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip().lower()
        if len(v) < 3:
            raise ValueError("Username must be at least 3 characters long.")
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Username can only contain letters, numbers, underscores, and hyphens.")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters long.")
        return v


class UserLogin(BaseModel):
    username_or_email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    full_name: Optional[str] = None
    institution: Optional[str] = None
    department: Optional[str] = None
    academic_year: Optional[str] = None
    research_interests: Optional[str] = None
    is_active: bool = True
    created_at: datetime


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    institution: Optional[str] = None
    department: Optional[str] = None
    academic_year: Optional[str] = None
    research_interests: Optional[str] = None


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    email: str
    reset_token: str
    new_password: str


class SavedQueryCreate(BaseModel):
    question: str
    answer: str
    domain: Optional[str] = None
    citations_json: str
    why_this_answer_json: str
    confidence: str


class SavedQueryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    question: str
    answer: str
    domain: Optional[str] = None
    citations_json: str
    why_this_answer_json: str
    confidence: str
    created_at: datetime
