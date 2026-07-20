"""Application configuration classes.

All values are read from the environment (loaded from ``.env`` by the factory
via python-dotenv). Non-secret defaults are provided so the app can boot in
development, but production must supply real secrets through the environment.
"""

import os
from datetime import timedelta


def _bool_env(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in {"1", "true", "yes", "on"}


class BaseConfig:
    """Settings shared by every environment."""

    # --- Core secrets ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-change-me")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-insecure-jwt-change-me")

    # --- Database ---
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "mysql+pymysql://user:password@localhost:3306/plantvest"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- JWT lifetimes (per BUILD_SPEC section 3) ---
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)

    # --- Redis / Celery broker (wired in Phase 7) ---
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    # --- File uploads (used from Phase 5) ---
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

    # --- CORS (locked down in Phase 11) ---
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")


class DevConfig(BaseConfig):
    DEBUG = True


class ProdConfig(BaseConfig):
    DEBUG = False
    TESTING = False


class TestConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    # Prefer an isolated test DB; fall back to in-memory SQLite so tests can run
    # without a MySQL instance.
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "TEST_DATABASE_URL", "sqlite:///:memory:"
    )


# Factory looks the requested environment up here.
config_map = {
    "development": DevConfig,
    "production": ProdConfig,
    "testing": TestConfig,
}
