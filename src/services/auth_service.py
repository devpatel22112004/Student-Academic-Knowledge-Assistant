import hashlib

# This module provides authentication and registration functions for an in-memory user management system. It includes password hashing, user validation, and account creation logic. The users are stored in a dictionary with email as the key and user details as the value.
def hash_password(password):
    """Hash a password before storing it in session memory."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

# The authenticate_user function checks if the provided email and password match a user in the users dictionary. The register_user function creates a new user after validating the input and ensuring the email is not already registered. Both functions return a standardized response indicating success or failure along with a message and user details if applicable.
def authenticate_user(users, email, password):
    """Validate login credentials against the in-memory users dictionary."""
    email_key = email.strip().lower()
    user = users.get(email_key)

    if not user:
        return {"ok": False, "message": "User not found. Please register first.", "user": None}

    if user["password"] != hash_password(password):
        return {"ok": False, "message": "Invalid password.", "user": None}

    return {"ok": True, "message": f"Welcome back, {user['name']}!", "user": user}

# The register_user function creates a new user after validating the input and ensuring the email is not already registered. Both functions return a standardized response indicating success or failure along with a message and user details if applicable.
def register_user(users, name, email, password, confirm_password):
    """Create a new in-memory user record after validation."""
    email_key = email.strip().lower()

    if not name.strip() or not email_key or not password:
        return {"ok": False, "message": "Name, email, and password are required.", "user": None}

    if password != confirm_password:
        return {"ok": False, "message": "Passwords do not match.", "user": None}

    if email_key in users:
        return {"ok": False, "message": "This email is already registered.", "user": None}

    user = {
        "name": name.strip(),
        "email": email_key,
        "password": hash_password(password),
    }
    users[email_key] = user

    return {"ok": True, "message": "Account created successfully.", "user": user}