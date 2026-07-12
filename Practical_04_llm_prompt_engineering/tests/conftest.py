"""
tests/conftest.py
===================

WHY THIS FILE EXISTS
----------------------
Auto-loaded by pytest before any test in tests/ runs; fixtures defined
here are available by name to every test function with no import needed.

Both test_variants.py and test_roles.py need to call code that ultimately
calls `client.converse(...)`. Rather than each test file mocking boto3
independently (and risking two slightly different fake response shapes),
this file provides ONE fake client fixture both files share — so if the
real Converse response shape ever changes, there's exactly one fixture to
update, not two.

It also clears `get_bedrock_runtime_client`'s @lru_cache before every test
(autouse=True) — without this, a client mocked/patched in one test could
leak into the next test via the cache, since @lru_cache persists across
calls within the same test process.
"""

import pytest


@pytest.fixture
def mock_bedrock_runtime_client(mocker):

    fake_client = mocker.MagicMock()
    fake_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Category: IT Support"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 40, "outputTokens": 6, "totalTokens": 46},
    }
    return fake_client


@pytest.fixture(autouse=True)
def _clear_client_cache():
    """
    Runs automatically before every test. Clears the @lru_cache on
    get_bedrock_runtime_client so each test starts from a clean slate
    instead of potentially reusing a mock (or real client) left behind by
    a previous test in the same pytest process.
    """
    from utils.config import get_bedrock_runtime_client

    get_bedrock_runtime_client.cache_clear()
    yield
    get_bedrock_runtime_client.cache_clear()
