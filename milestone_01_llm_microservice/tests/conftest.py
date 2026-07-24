"""
conftest.py
-----------
Shared pytest fixtures. Critically, we never let tests hit real AWS Bedrock:
`app.dependency_overrides` swaps in a fake BedrockService, so the suite is
fast, free, and deterministic instead of depending on live credentials.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.bedrock_service import get_bedrock_service
from app.utils.exceptions import BedrockError


class FakeBedrockService:
    """Stand-in for BedrockService that returns canned results, no network calls."""

    async def classify(self, text: str) -> tuple[str, float]:
        if not text.strip():
            raise BedrockError(message="Empty text cannot be classified")
        return "technology", 0.87

    async def summarize(self, text: str, max_length: int) -> str:
        if not text.strip():
            raise BedrockError(message="Empty text cannot be summarized")
        return " ".join(text.split()[:max_length]) or "summary"


@pytest.fixture
def client() -> TestClient:
    """Returns a TestClient with the real BedrockService swapped for the fake one."""
    app.dependency_overrides[get_bedrock_service] = lambda: FakeBedrockService()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
