"""
Authentication helpers for admin-protected routes.
"""

from typing import Optional

from jose import JWTError, jwt

from cyber_inference.core.config import get_settings
from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)


def is_admin_password_set() -> bool:
    """Return True when admin protection is enabled."""
    return bool(get_settings().admin_password)


def extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Extract the bearer token from an Authorization header value."""
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    return authorization.split(" ", 1)[1].strip() or None


def verify_admin_token_value(token: Optional[str]) -> bool:
    """Validate an admin JWT token."""
    settings = get_settings()
    if not settings.admin_password:
        return True
    if not token:
        return False

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
        )
    except JWTError as exc:
        logger.debug(f"Invalid admin token: {exc}")
        return False

    return payload.get("type") == "admin"
