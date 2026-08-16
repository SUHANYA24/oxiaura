"""Application factory for the PlantVest AI backend."""

import os

from dotenv import load_dotenv
from flask import Flask, jsonify

from .config import config_map
from .extensions import celery_init_app, cors, db, jwt, ma, migrate

# Load environment variables from .env as early as possible.
load_dotenv()

API_PREFIX = "/api/v1"


def create_app(config_name: str | None = None) -> Flask:
    """Build and configure a Flask application instance.

    :param config_name: one of ``development`` / ``production`` / ``testing``.
        Falls back to the ``APP_ENV`` env var, then ``development``.
    """
    if config_name is None:
        config_name = os.environ.get("APP_ENV", "development")
    config_class = config_map.get(config_name, config_map["development"])

    app = Flask(__name__)
    app.config.from_object(config_class)

    _init_extensions(app)
    _register_blueprints(app)

    return app


def _init_extensions(app: Flask) -> None:
    db.init_app(app)
    # Import models so their tables register on ``db.metadata`` before
    # Flask-Migrate inspects it for autogeneration.
    from . import models  # noqa: F401

    migrate.init_app(app, db)
    _init_jwt(app)
    ma.init_app(app)
    cors.init_app(app)

    # Celery (Phase 7): create the app-bound Celery instance and import the task
    # module so ``@shared_task`` definitions register on it.
    celery_init_app(app)
    from .tasks import celery_tasks  # noqa: F401


def _init_jwt(app: Flask) -> None:
    """Initialize JWT and register the blocklist check + JSON error handlers.

    Access/refresh lifetimes come from the config (15 min / 7 days). The
    blocklist loader is what enforces logout/revocation on every request.
    """
    jwt.init_app(app)

    from .utils.token_blocklist import is_blocklisted

    @jwt.token_in_blocklist_loader
    def _token_revoked(_jwt_header, jwt_payload) -> bool:
        return is_blocklisted(jwt_payload["jti"])

    # Consistent JSON shapes for the various auth failure modes. Missing,
    # malformed, expired, and revoked tokens all resolve to 401; role checks
    # (403) are handled by ``role_required``.
    @jwt.unauthorized_loader
    def _missing_token(reason):
        return jsonify({"error": "authorization_required", "message": reason}), 401

    @jwt.invalid_token_loader
    def _invalid_token(reason):
        return jsonify({"error": "invalid_token", "message": reason}), 401

    @jwt.expired_token_loader
    def _expired_token(_jwt_header, _jwt_payload):
        return jsonify({"error": "token_expired", "message": "Token has expired."}), 401

    @jwt.revoked_token_loader
    def _revoked_token(_jwt_header, _jwt_payload):
        return jsonify({"error": "token_revoked", "message": "Token has been revoked."}), 401

    @jwt.needs_fresh_token_loader
    def _needs_fresh_token(_jwt_header, _jwt_payload):
        return jsonify({"error": "fresh_token_required", "message": "Fresh token required."}), 401


def _register_blueprints(app: Flask) -> None:
    from .routes.auth import auth_bp
    from .routes.customers import customers_bp
    from .routes.documents import documents_bp
    from .routes.health import health_bp

    app.register_blueprint(health_bp, url_prefix=API_PREFIX)
    app.register_blueprint(auth_bp, url_prefix=API_PREFIX)
    app.register_blueprint(customers_bp, url_prefix=API_PREFIX)
    app.register_blueprint(documents_bp, url_prefix=API_PREFIX)
