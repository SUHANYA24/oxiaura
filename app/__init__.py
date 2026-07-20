"""Application factory for the PlantVest AI backend."""

import os

from dotenv import load_dotenv
from flask import Flask

from .config import config_map
from .extensions import cors, db, jwt, ma, migrate

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
    jwt.init_app(app)
    ma.init_app(app)
    cors.init_app(app)


def _register_blueprints(app: Flask) -> None:
    from .routes.health import health_bp

    app.register_blueprint(health_bp, url_prefix=API_PREFIX)
