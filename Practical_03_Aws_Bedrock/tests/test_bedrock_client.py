"""
tests/test_bedrock_client.py
==============================
Unit tests for src/bedrock_client.py — the client-factory module.

What we're actually verifying here:
    1. Calling get_bedrock_client() / get_bedrock_runtime_client() results
       in boto3.client() being called with the correct service name
       ("bedrock" vs "bedrock-runtime") — this is the one thing that, if
       wrong, would silently break every other module in the project.
    2. The result is cached (same object returned on repeated calls),
       proving the @lru_cache is actually working.
    3. Missing credentials produce our friendly RuntimeError, not a raw
       botocore traceback.
"""

from botocore.exceptions import NoCredentialsError

from src.bedrock_client import get_bedrock_client, get_bedrock_runtime_client


def test_get_bedrock_client_uses_correct_service_name(mocker):
    """The control-plane client must be built with service name 'bedrock'."""
    mock_boto_client = mocker.patch("src.bedrock_client.boto3.client")

    get_bedrock_client()

    mock_boto_client.assert_called_once()
    called_service_name = mock_boto_client.call_args[0][0]
    assert called_service_name == "bedrock"


def test_get_bedrock_runtime_client_uses_correct_service_name(mocker):
    """The data-plane client must be built with service name 'bedrock-runtime'."""
    mock_boto_client = mocker.patch("src.bedrock_client.boto3.client")

    get_bedrock_runtime_client()

    mock_boto_client.assert_called_once()
    called_service_name = mock_boto_client.call_args[0][0]
    assert called_service_name == "bedrock-runtime"


def test_client_is_cached_across_calls(mocker):
    """
    Calling get_bedrock_client() twice should only construct boto3.client
    ONCE — the second call should return the cached object.
    """
    mock_boto_client = mocker.patch("src.bedrock_client.boto3.client")

    first = get_bedrock_client()
    second = get_bedrock_client()

    assert first is second
    mock_boto_client.assert_called_once()


def test_missing_credentials_raise_friendly_error(mocker):
    """
    If boto3 can't find credentials, we should surface a clear RuntimeError
    with setup instructions — not let the raw NoCredentialsError propagate.
    """
    mocker.patch(
        "src.bedrock_client.boto3.client", side_effect=NoCredentialsError()
    )

    try:
        get_bedrock_client()
        assert False, "Expected RuntimeError to be raised"
    except RuntimeError as exc:
        assert "aws configure" in str(exc)
