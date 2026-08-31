"""Celery worker entry point (BUILD_SPEC Phase 7).

Boots a Flask application (so tasks have the ORM, config, and services) and
exposes the app-bound Celery instance for the worker to import.

Run the worker (needs a reachable Redis via ``REDIS_URL``):

    celery -A celery_worker.celery_app worker --loglevel=info

On Windows, add ``--pool=solo`` (or ``--pool=threads``) since the default
prefork pool is not supported there:

    celery -A celery_worker.celery_app worker --loglevel=info --pool=solo
"""

from app import create_app

flask_app = create_app()
celery_app = flask_app.extensions["celery"]
