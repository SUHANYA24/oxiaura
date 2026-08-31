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


def _celery_settings(redis_url: str, eager: bool) -> dict:
    """Celery settings for a real broker, or for running with no broker at all.

    ``eager`` swaps Redis out for in-process execution: ``.delay()`` runs the task
    on the calling thread and returns a finished result. Nothing connects to
    Redis, so no ``redis-server`` and no ``celery worker`` are needed — which is
    what makes the app runnable on a machine without either.

    Both URLs have to move, not just the broker. ``.delay()`` skips the broker
    under ``task_always_eager``, but ``GET /documents/jobs/<task_id>`` builds an
    ``AsyncResult``, and that reads the *result backend* — leave it pointing at
    Redis and job polling still raises ConnectionError.

    The cost is that OCR + fraud then run inside the HTTP request. On CPU that is
    seconds to tens of seconds for the first upload (EasyOCR loads its models),
    and the request blocks for all of it.
    """
    if eager:
        return {
            "broker_url": "memory://",
            "result_backend": "cache+memory://",
            "task_always_eager": True,
            # False, so a failing task is recorded as FAILURE and polled for like
            # any other — rather than escaping into the upload response as a 500.
            # This keeps the endpoint's contract identical in both modes.
            "task_eager_propagates": False,
            "task_store_eager_result": True,
            "task_ignore_result": False,
            "task_track_started": True,
        }
    return {
        "broker_url": redis_url,
        "result_backend": redis_url,
        "task_ignore_result": False,
        "task_track_started": True,
        "broker_connection_retry_on_startup": True,
    }


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

    # CELERY_EAGER=1 runs OCR + fraud in-process instead of on a worker, with no
    # Redis involved. Set it when you have no broker running: without it,
    # ``POST /documents/upload`` stores the file and then fails with
    # "Error 10061 connecting to localhost:6379" from ``process_document.delay``,
    # leaving a document row behind that nothing will ever process.
    CELERY_EAGER = _bool_env("CELERY_EAGER", False)

    # Celery moves OCR + fraud off the request thread. Broker and result backend
    # both use Redis; task results are kept so the job-status endpoint can poll.
    CELERY = _celery_settings(REDIS_URL, CELERY_EAGER)

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
    # Production always uses a real broker, whatever CELERY_EAGER says. Eager mode
    # would run a multi-second OCR pass inside the HTTP request, holding a worker
    # thread per upload and timing clients out under any real load.
    CELERY_EAGER = False
    CELERY = _celery_settings(BaseConfig.REDIS_URL, eager=False)


class TestConfig(BaseConfig):
    TESTING = True
    DEBUG = True
    # Prefer an isolated test DB; fall back to in-memory SQLite so tests can run
    # without a MySQL instance.
    SQLALCHEMY_DATABASE_URI = _str_env("TEST_DATABASE_URL", "sqlite:///:memory:")

    # Always eager, ignoring CELERY_EAGER: the suite must not depend on a broker.
    # Unlike dev, exceptions propagate, so a task that breaks fails its test
    # loudly instead of hiding in a result the test never polls.
    CELERY_EAGER = True
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
