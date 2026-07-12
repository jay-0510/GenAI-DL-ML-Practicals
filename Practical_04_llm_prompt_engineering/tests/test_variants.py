"""
tests/test_variants.py
========================
Unit tests for src/variants.py.

What we verify:
    1. Each prompt builder (zero-shot/few-shot/CoT) produces text with the
       distinguishing features that make it that technique — e.g. few-shot
       must contain worked examples, CoT must ask for step-by-step reasoning.
       This matters because if build_few_shot_prompt() ever regressed to
       look like build_zero_shot_prompt(), the "comparison" in the notebook
       would silently become meaningless (comparing a technique to itself).
    2. call_bedrock() builds the correct Converse request shape.
    3. classify_ticket() rejects unknown variants instead of silently
       defaulting.
    4. compare_variants() calls all three techniques and labels each result.
"""

import pytest

from src.variants import (
    CATEGORIES,
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_cot_prompt,
    call_bedrock,
    classify_ticket,
    compare_variants,
)


def test_zero_shot_prompt_has_no_examples_or_reasoning_scaffold():
    prompt = build_zero_shot_prompt("My VPN isn't connecting.")
    assert "Ticket:" in prompt
    for category in CATEGORIES:
        assert category in prompt
    # Distinguishing feature: no worked examples, no "step by step" ask.
    assert "Reasoning" not in prompt
    assert prompt.count("Ticket:") == 1  # only the real ticket, no examples


def test_few_shot_prompt_contains_worked_examples():
    prompt = build_few_shot_prompt("My VPN isn't connecting.")
    # Distinguishing feature: multiple "Ticket: ... Category: ..." pairs
    # (the worked examples) PLUS the real ticket at the end.
    assert prompt.count("Category:") >= 4  # 3 examples + final answer slot
    assert "casual leave" in prompt  # one of the seeded examples


def test_cot_prompt_asks_for_step_by_step_reasoning():
    prompt = build_cot_prompt("My VPN isn't connecting.")
    assert "step by step" in prompt.lower()
    assert "Reasoning:" in prompt
    assert "Category:" in prompt


def test_call_bedrock_builds_correct_request_shape(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )

    result = call_bedrock(user_prompt="classify this", temperature=0.2, top_p=0.8, max_tokens=64)

    call_kwargs = mock_bedrock_runtime_client.converse.call_args.kwargs
    assert call_kwargs["messages"][0]["content"][0]["text"] == "classify this"
    assert call_kwargs["inferenceConfig"] == {
        "temperature": 0.2,
        "topP": 0.8,
        "maxTokens": 64,
    }
    assert "system" not in call_kwargs  # no system_prompt was passed
    assert result["text"] == "Category: IT Support"


def test_call_bedrock_includes_system_prompt_when_given(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )

    call_bedrock(user_prompt="hi", system_prompt="You are terse.")

    call_kwargs = mock_bedrock_runtime_client.converse.call_args.kwargs
    assert call_kwargs["system"] == [{"text": "You are terse."}]


def test_classify_ticket_rejects_unknown_variant(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )
    with pytest.raises(ValueError):
        classify_ticket("some ticket", variant="not_a_real_variant")


def test_classify_ticket_tags_result_with_variant_name(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )
    result = classify_ticket("some ticket", variant="cot")
    assert result["variant"] == "cot"


def test_compare_variants_runs_all_three_techniques(mocker, mock_bedrock_runtime_client):
    mocker.patch(
        "src.variants.get_bedrock_runtime_client",
        return_value=mock_bedrock_runtime_client,
    )
    results = compare_variants("some ticket")
    assert set(results.keys()) == {"zero_shot", "few_shot", "cot"}
    assert mock_bedrock_runtime_client.converse.call_count == 3
