"""Shared service-layer exceptions.

Services raise these to signal an outcome; routes catch ``ServiceError`` and
translate it into a consistent JSON error response with the carried status code.
This keeps HTTP concerns out of the service layer.
"""


class ServiceError(Exception):
    """Base class for expected, user-facing service failures."""

    status = 400
    error = "service_error"

    def __init__(self, message: str, *, status: int | None = None, error: str | None = None):
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status
        if error is not None:
            self.error = error


class NotFoundError(ServiceError):
    status = 404
    error = "not_found"


class ConflictError(ServiceError):
    """A uniqueness/business-rule conflict (e.g. duplicate NIC)."""

    status = 409
    error = "conflict"


class ForbiddenError(ServiceError):
    status = 403
    error = "forbidden"


class ValidationError(ServiceError):
    """Business-rule validation failure not caught by the schema."""

    status = 422
    error = "validation_error"


class UnsupportedMediaError(ServiceError):
    """An uploaded file has a disallowed / unrecognized media type."""

    status = 415
    error = "unsupported_media_type"
