"""
tests/test_roles.py
=====================
Unit tests for src/roles.py — the persona/system-message experiments.

What we verify:
    1. get_system_prompt() returns the right text per role, and raises on
       an unknown role instead of silently returning an empty string.
    2. run_with_role() passes the persona text as the Converse `system`
       field, and passes the SAME user prompt regardless of role (this is
       the property the whole experiment depends on — if the user prompt
       changed per role, tone differences couldn't be attributed to the
       system message alone).
    3. compare_roles() runs every persona and tags each result with its
       role name.

Note: we patch `src.variants.get_bedrock_runtime_client` (not
`src.roles.get_bedrock_runtime_client`) because roles.py calls through to
variants.call_bedrock(), which resolves the client from within the
variants module's namespace. This is exactly the kind of detail a shared
low-level call function makes easy to get right consistently.
"""

import pytest

from src.roles import SYSTEM_PROMPTS, get_system_prompt, run_with_role, compare_roles


def test_get_system_prompt_returns_correct_text_per_role():
    assert get_system_prompt("senior_data_analyst") == SYSTEM_PROMPTS["senior_data_analyst"]
    assert get_system_prompt("creative_writer") == SYSTEM_PROMPTS["creative_writer"]


def test_get_system_prompt_raises_on_unknown_role():
    with pytest.raises(ValueError):
        get_system_prompt("not_a_real_role")


def test_run_with_role_sends_persona_as_system_message(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )

    run_with_role("Explain employee turnover.", role="creative_writer")

    call_kwargs = mock_bedrock_runtime_client.converse.call_args.kwargs
    assert call_kwargs["system"] == [{"text": SYSTEM_PROMPTS["creative_writer"]}]
    assert call_kwargs["messages"][0]["content"][0]["text"] == "Explain employee turnover."


def test_run_with_role_tags_result_with_role_name(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )
    result = run_with_role("Explain employee turnover.", role="senior_data_analyst")
    assert result["role"] == "senior_data_analyst"


def test_compare_roles_uses_identical_prompt_across_roles(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )

    compare_roles("Explain employee turnover.")

    prompts_sent = [
        call.kwargs["messages"][0]["content"][0]["text"]
        for call in mock_bedrock_runtime_client.converse.call_args_list
    ]
    # Every call must have used the exact same user prompt — only the
    # system message should differ between roles.
    assert len(set(prompts_sent)) == 1


def test_compare_roles_returns_all_personas_by_default(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )
    results = compare_roles("Explain employee turnover.")
    assert set(results.keys()) == set(SYSTEM_PROMPTS.keys())
