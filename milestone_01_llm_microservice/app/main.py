"""
main.py
-------
FastAPI application entry point. Responsible only for assembling the app:
metadata, routers, and exception handlers. No business logic lives here.

Run with: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging

from fastapi import FastAPI

from app.config import get_settings
from app.routes import classify, summarize
from app.utils.exceptions import register_exception_handlers

settings = get_settings()
logging.basicConfig(level=settings.log_level)

app = FastAPI(
    title="LLM Microservice",
    description="FastAPI microservice backed by AWS Bedrock for text classification and summarization.",
    version="1.0.0",
)

# Routers are included with no prefix since the README specifies bare
# /classify and /summarize paths (not /api/v1/... etc.).
app.include_router(classify.router)
app.include_router(summarize.router)

register_exception_handlers(app)


@app.get("/", tags=["health"], summary="Health check")
async def root() -> dict:
    """Basic liveness endpoint — confirms the service is up and reachable."""
    return {"status": "ok", "service": "llm-microservice", "env": settings.app_env}
