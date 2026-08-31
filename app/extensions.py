"""Flask extension instances.

Instantiated here without an app so they can be imported anywhere (models,
services, routes) without causing circular imports. Each is bound to the
application inside the ``create_app`` factory via ``init_app``.
"""

from celery import Celery, Task
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager
from flask_marshmallow import Marshmallow
from flask_cors import CORS

# ORM / database
db = SQLAlchemy()

# Alembic migrations
migrate = Migrate()

# JWT auth (configured in the factory in Phase 3)
jwt = JWTManager()

# Marshmallow serialization / validation
ma = Marshmallow()

# Cross-origin resource sharing
cors = CORS()


def celery_init_app(app: Flask) -> Celery:
    """Create the Celery app and bind it to Flask (BUILD_SPEC Phase 7).

    Tasks run inside a Flask application context so they can use the ORM,
    services, and ``current_app`` config exactly as request handlers do. The
    Celery configuration (broker, result backend, eager mode for tests) is read
    from ``app.config['CELERY']``. The instance is stashed on
    ``app.extensions['celery']`` and set as the default so ``@shared_task``
    definitions bind to it.
    """

    class FlaskTask(Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery_app = Celery(app.name, task_cls=FlaskTask)
    celery_app.config_from_object(app.config["CELERY"])
    celery_app.set_default()
    app.extensions["celery"] = celery_app
    return celery_app
