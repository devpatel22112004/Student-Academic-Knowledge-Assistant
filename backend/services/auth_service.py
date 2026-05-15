"""Authentication service for user registration and login."""

import hashlib
import jwt
from datetime import datetime, timedelta
from backend.models import User
from backend.config import JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRATION


def hash_password(password: str) -> str:
    """Hash password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return hash_password(password) == hashed


def generate_token(user_id: str) -> str:
    """Generate JWT token for user."""
    payload = {
        "user_id": str(user_id),
        "exp": datetime.utcnow() + JWT_EXPIRATION,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """Verify JWT token and extract user_id."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")


def register_user(name: str, email: str, password: str) -> dict:
    """Register a new user.
    
    Args:
        name: User's full name
        email: User's email
        password: Plaintext password
        
    Returns:
        dict with user_id and token
        
    Raises:
        ValueError: If email already exists
    """
    email = email.lower().strip()
    
    # Check if user exists
    existing = User.find_by_email(email)
    if existing:
        raise ValueError(f"User with email {email} already exists")
    
    # Create user
    hashed = hash_password(password)
    user = User.create(name=name, email=email, hashed_password=hashed)
    
    # Generate token
    token = generate_token(user["_id"])
    
    return {
        "user_id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "token": token,
    }


def login_user(email: str, password: str) -> dict:
    """Login user and return token.
    
    Args:
        email: User's email
        password: Plaintext password
        
    Returns:
        dict with user_id, name, and token
        
    Raises:
        ValueError: If credentials are invalid
    """
    email = email.lower().strip()
    user = User.find_by_email(email)
    
    if not user:
        raise ValueError("Invalid email or password")
    
    if not verify_password(password, user["password"]):
        raise ValueError("Invalid email or password")
    
    # Generate token
    token = generate_token(user["_id"])
    
    return {
        "user_id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "token": token,
    }


def get_user_profile(user_id: str) -> dict:
    """Get user profile information."""
    user = User.find_by_id(user_id)
    if not user:
        raise ValueError("User not found")
    
    return {
        "user_id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "created_at": user["created_at"].isoformat(),
    }
