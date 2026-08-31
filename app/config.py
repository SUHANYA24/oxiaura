"""Application configuration classes.

All values are read from the environment (loaded from ``.env`` by the factory
via python-dotenv). Non-secret defaults are provided so the app can boot in
development, but production must supply real secrets through the environment.
"""

import os
from datetime import timedelta


def _bool_env(key: str, default: bool = False) -> bool:
    return os.environ.get(key, str(default)).lower() in {"1", "true", "yes", "on"}


def _str_env(key: str, default: str) -> str:
    """Return ``key`` from the environment, treating blank as unset.

    ``.env`` files commonly carry placeholder lines such as ``TEST_DATABASE_URL=``.
    ``os.environ.get`` would hand back ``""`` for those, which then fails deep
    inside the consumer (e.g. SQLAlchemy: "Could not parse SQLAlchemy URL from
    string ''"). Blank values fall back to the default instead.
    """
    value = os.environ.get(key)
    return value.strip() if value and value.strip() else default


class BaseConfig:
    """Settings shared by every environment."""

    # --- Core secrets ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-insecure-change-me")
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-insecure-jwt-change-me")
    # Signs agreement QR tokens (Phase 8). Falls back to SECRET_KEY so dev boots.
    QR_SECRET_KEY = os.environ.get("QR_SECRET_KEY") or os.environ.get(
        "SECRET_KEY", "dev-insecure-change-me"
    )

    # --- Database ---
    SQLALCHEMY_DATABASE_URI = _str_env(
        "DATABASE_URL", "mysql+pymysql://user:password@localhost:3306/plantvest"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- JWT lifetimes (per BUILD_SPEC section 3) ---
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)

    # --- Redis / Celery broker (wired in Phase 7) ---
    REDIS_URL = _str_env("REDIS_URL", "redis://localhost:6379/0")

    # Celery moves OCR + fraud off the request thread. Broker and result backend
    # both use Redis; task results are kept so the job-status endpoint can poll.
    CELERY = {
        "broker_url": REDIS_URL,
        "result_backend": REDIS_URL,
        "task_ignore_result": False,
        "task_track_started": True,
        "broker_connection_retry_on_startup": True,
    }

    # --- File uploads (used from Phase 5) ---
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "10"))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024

    # --- Generated agreement PDFs (Phase 8) ---
    AGREEMENTS_FOLDER = os.environ.get("AGREEMENTS_FOLDER", "agreements")

    # --- Fraud detection (Phase 6) ---
    # A document is flagged when its weighted aggregate score exceeds this.
    FRAUD_FLAG_THRESHOLD = float(os.environ.get("FRAUD_FLAG_THRESHOLD", "60"))
    # Optional trained-model weights. When unset, the CNN/Siamese detectors fall
    # back to deterministic mock scores (BUILD_SPEC Phase 6 note; real weights
    # are swapped in at Phase 6b).
    FRAUD_CNN_WEIGHTS = os.environ.get("FRAUD_CNN_WEIGHTS") or None
    FRAUD_SIAMESE_WEIGHTS = os.environ.get("FRAUD_SIAMESE_WEIGHTS") or None
    # Bank of known-forgery embeddings built by ml_training.build_reference_bank.
    # The Siamese detector needs BOTH this and FRAUD_SIAMESE_WEIGHTS to leave its
    # mock path: weights alone give it nothing to compare an upload against.
    FRAUD_REFERENCE_BANK = os.environ.get("FRAUD_REFERENCE_BANK") or None

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
    SQLALCHEMY_DATABASE_URI = _str_env("TEST_DATABASE_URL", "sqlite:///:memory:")

    # Run Celery tasks synchronously in-process — no Redis broker/worker needed.
    # Eager results are stored in an in-memory backend so the job-status
    # endpoint can still be exercised, and exceptions propagate to the caller.
    CELERY = {
        "broker_url": "memory://",
        "result_backend": "cache+memory://",
        "task_always_eager": True,
        "task_eager_propagates": True,
        "task_store_eager_result": True,
    }


# Factory looks the requested environment up here.
config_map = {
    "development": DevConfig,
    "production": ProdConfig,
    "testing": TestConfig,
}
