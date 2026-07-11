"""
tests/conftest.py
===================

WHAT THIS FILE DOES (and why every pytest project has one)
-------------------------------------------------------------
`conftest.py` is a special filename that pytest auto-discovers and loads
BEFORE running any tests in this directory — you never import it manually.
Any fixture defined here becomes automatically available, by name, as an
argument to ANY test function in tests/, with no import statement needed.

Its job here is to hold everything that would otherwise be DUPLICATED
across test_bedrock_client.py and test_converse.py:

    1. A fake ("mocked") Bedrock client, so tests never make a real network
       call to AWS. This matters because:
         - Tests must be able to run with zero AWS credentials (e.g. in a
           CI pipeline that has no AWS access).
         - Tests must be fast and free — a real Bedrock call costs money
           and takes ~1-2 seconds; a mock is instant and free.
         - Tests must be deterministic — a real LLM call can return a
           different response every time, which breaks assertions.

    2. Sample request/response payloads shaped exactly like what the real
       Bedrock Converse API returns, so both test files can assert against
       realistic data without each hand-rolling their own fixture.

Think of conftest.py as the "test fixtures warehouse" for this whole
directory — bedrock_client.py is to src/ what conftest.py is to tests/:
a single shared place other files depend on instead of re-implementing.
"""

import pytest


@pytest.fixture
def mock_bedrock_runtime_client(mocker):
    """
    Returns a MagicMock standing in for the real boto3 bedrock-runtime
    client, with `.converse()` pre-programmed to return a realistic fake
    response.

    Why build the fake response this way instead of calling real AWS once
    and hardcoding the captured JSON?
        Building it programmatically keeps the fixture readable and makes
        it trivial for a future test to override just one field (e.g.
        stopReason) without maintaining a giant hardcoded JSON blob.

    `mocker` here comes from pytest-mock (a pytest wrapper around
    unittest.mock) — it's what lets us swap out the real
    `get_bedrock_runtime_client` for this fake one during a test.
    """
    fake_client = mocker.MagicMock()
    fake_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "This is a mocked Bedrock response."}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 12, "outputTokens": 8, "totalTokens": 20},
    }
    return fake_client


@pytest.fixture
def mock_bedrock_control_client(mocker):
    """
    Returns a MagicMock standing in for the real boto3 'bedrock' (control
    plane) client, with `.list_foundation_models()` pre-programmed to
    return two fake Amazon Nova models. This allows tests to exercise the
    provider_filter logic without making real AWS API calls.
    """
    fake_client = mocker.MagicMock()
    fake_client.list_foundation_models.return_value = {
        "modelSummaries": [
            {
                "modelId": "amazon.nova-micro-v1:0",
                "providerName": "Amazon",
                "modelName": "Nova Micro",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
            },
            {
                "modelId": "amazon.nova-lite-v1:0",
                "providerName": "Amazon",
                "modelName": "Nova Lite",
                "inputModalities": ["TEXT"],
                "outputModalities": ["TEXT"],
            },
        ]
    }
    return fake_client


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """
    Runs automatically (autouse=True) before EVERY test in this directory.

    Why is this necessary?
        get_bedrock_client() / get_bedrock_runtime_client() in
        src/bedrock_client.py are wrapped with @lru_cache so the *real*
        app only builds one client per process. But that caching is
        actively HARMFUL in tests: if test A patches the client and test B
        runs afterwards expecting a fresh mock, the lru_cache would hand
        test B test A's stale cached object. This fixture clears that
        cache before each test so every test starts from a clean slate.
    """
    from src.bedrock_client import get_bedrock_client, get_bedrock_runtime_client

    get_bedrock_client.cache_clear()
    get_bedrock_runtime_client.cache_clear()
    yield
    get_bedrock_client.cache_clear()
    get_bedrock_runtime_client.cache_clear()
