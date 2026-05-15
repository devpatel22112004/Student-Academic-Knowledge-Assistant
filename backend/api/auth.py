"""Authentication API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from backend.services import auth_service

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterRequest(BaseModel):
    """User registration request."""
    name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    """Authentication response."""
    user_id: str
    name: str
    email: str
    token: str


@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """Register a new user."""
    try:
        result = auth_service.register_user(
            name=req.name,
            email=req.email,
            password=req.password,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Login user and return token."""
    try:
        result = auth_service.login_user(
            email=req.email,
            password=req.password,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.get("/profile")
async def get_profile(user_id: str):
    """Get user profile."""
    try:
        return auth_service.get_user_profile(user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
