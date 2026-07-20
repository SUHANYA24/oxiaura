"""Flask extension instances.

Instantiated here without an app so they can be imported anywhere (models,
services, routes) without causing circular imports. Each is bound to the
application inside the ``create_app`` factory via ``init_app``.
"""

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
