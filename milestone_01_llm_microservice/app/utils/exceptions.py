"""
exceptions.py
-------------
Custom exception types + their FastAPI handlers, all in one file so the
mapping between "thing that went wrong" and "HTTP response shape" is easy
to audit in one place instead of scattered try/excepts per route.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.models.schemas import ErrorResponse, ErrorDetail

logger = logging.getLogger("llm_microservice")


class AppError(Exception):
    """Base class for all app-raised errors, carries an HTTP status + code."""

    def __init__(self, message: str, code: str, status_code: int, details: str | None = None):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().__init__(message)


class ValidationError(AppError):
    """Raised for request-shape problems that slip past Pydantic (400)."""

    def __init__(self, message: str, details: str | None = None):
        super().__init__(message, code="VALIDATION_ERROR", status_code=400, details=details)


class AuthenticationError(AppError):
    """Raised when AWS credentials are missing/invalid (401)."""

    def __init__(self, message: str = "AWS authentication failed", details: str | None = None):
        super().__init__(message, code="AUTHENTICATION_ERROR", status_code=401, details=details)


class BedrockError(AppError):
    """Raised when AWS Bedrock itself fails or is unreachable (503)."""

    def __init__(self, message: str = "Bedrock service unavailable", details: str | None = None):
        super().__init__(message, code="BEDROCK_ERROR", status_code=503, details=details)


def _error_body(code: str, message: str, details: str | None = None) -> dict:
    """Builds the standard {error: {...}} envelope so every handler agrees on shape."""
    return ErrorResponse(error=ErrorDetail(code=code, message=message, details=details)).model_dump()


def register_exception_handlers(app: FastAPI) -> None:
    """
    Wires all exception handlers onto the FastAPI app.
    Why a registration function instead of decorators in main.py: keeps
    main.py focused on app assembly, not error-handling logic.
    """

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        """Catches all our custom AppError subclasses (Validation/Auth/Bedrock)."""
        logger.warning("AppError: %s (%s)", exc.message, exc.code)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message, exc.details),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_pydantic_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Catches FastAPI/Pydantic's own 422 validation errors and reshapes them
        into the same error envelope, so clients only ever deal with one error format."""
        logger.warning("RequestValidationError: %s", exc.errors())
        first = exc.errors()[0] if exc.errors() else {}
        return JSONResponse(
            status_code=400,
            content=_error_body(
                "VALIDATION_ERROR",
                first.get("msg", "Invalid request payload"),
                details=str(exc.errors()),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Last-resort catch-all so an unhandled exception never leaks a raw
        traceback to the client; it still gets logged for debugging."""
        logger.exception("Unhandled exception")
        return JSONResponse(
            status_code=500,
            content=_error_body("INTERNAL_ERROR", "An unexpected error occurred"),
        )
