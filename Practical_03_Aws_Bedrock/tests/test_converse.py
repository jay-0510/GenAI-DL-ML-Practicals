"""
tests/test_converse.py
========================
Unit tests for src/converse.py — the Converse API wrapper.

What we're actually verifying here:
    1. send_prompt() builds the correct Converse request shape (messages,
       inferenceConfig with temperature/topP/maxTokens).
    2. send_prompt() correctly PARSES the nested Converse response back
       into our simple flat dict — this nested structure
       (output.message.content[0].text) is exactly the kind of thing that
       silently breaks if AWS changes response shape, so it deserves its
       own explicit test.
    3. An optional system_prompt is only included when provided.

All tests use `mock_bedrock_runtime_client` from conftest.py, so none of
them make a real network call.
"""

from src.converse import send_prompt


def test_send_prompt_returns_parsed_response(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.converse.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )

    result = send_prompt(prompt="Hello Bedrock")

    assert result["text"] == "This is a mocked Bedrock response."
    assert result["stop_reason"] == "end_turn"
    assert result["input_tokens"] == 12
    assert result["output_tokens"] == 8


def test_send_prompt_builds_correct_request_shape(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.converse.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )

    send_prompt(
        prompt="What is Bedrock?",
        model_id="amazon.nova-lite-v1:0",
        temperature=0.1,
        top_p=0.8,
        max_tokens=256,
    )

    call_kwargs = mock_bedrock_runtime_client.converse.call_args.kwargs
    assert call_kwargs["modelId"] == "amazon.nova-lite-v1:0"
    assert call_kwargs["messages"][0]["role"] == "user"
    assert call_kwargs["messages"][0]["content"][0]["text"] == "What is Bedrock?"
    assert call_kwargs["inferenceConfig"] == {
        "temperature": 0.1,
        "topP": 0.8,
        "maxTokens": 256,
    }
    # No system_prompt was passed, so 'system' should be absent entirely.
    assert "system" not in call_kwargs


def test_send_prompt_includes_system_prompt_when_given(
    mocker, mock_bedrock_runtime_client
):
    mocker.patch(
        "src.converse.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )

    send_prompt(prompt="Hi", system_prompt="You are a concise assistant.")

    call_kwargs = mock_bedrock_runtime_client.converse.call_args.kwargs
    assert call_kwargs["system"] == [{"text": "You are a concise assistant."}]


def test_send_prompt_uses_default_model_id_when_not_specified(
    mocker, mock_bedrock_runtime_client
):
    mocker.patch(
        "src.converse.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )

    from utils.config import config

    send_prompt(prompt="Hi")

    call_kwargs = mock_bedrock_runtime_client.converse.call_args.kwargs
    assert call_kwargs["modelId"] == config.default_model_id
