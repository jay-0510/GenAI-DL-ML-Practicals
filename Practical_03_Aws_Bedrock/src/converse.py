"""
src/converse.py
=================

WHAT THIS FILE DOES
--------------------
Satisfies practical task #2: "Invoke Claude or Titan via the Converse API.
Send a prompt and print the response," and task #3 (temperature / top_p
experimentation), by exposing one clean function: `send_prompt(...)`.

WHY THE CONVERSE API INSTEAD OF InvokeModel?
----------------------------------------------
Bedrock's older `invoke_model()` requires a DIFFERENT JSON request/response
body PER MODEL FAMILY — Anthropic's schema looks nothing like Titan's, which
looks nothing like Llama's. That means switching models means rewriting
your request-building code.

The `Converse` API (released 2024) is AWS's model-agnostic chat interface:
one request shape, one response shape, works across Claude, Titan, Llama,
Mistral, etc. Since this project's whole point is "invoke Claude OR Titan,"
Converse is the correct tool — it's literally what it was built for.
"""

from typing import Optional

from src.bedrock_client import get_bedrock_runtime_client


def send_prompt(
    prompt: str,
    model_id: Optional[str] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.5,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict:
    """
    Sends a single user prompt to a Bedrock foundation model via the
    Converse API and returns the parsed result.

    Args:
        prompt: The user's message/question.
        model_id: Bedrock model identifier, e.g.
            "anthropic.claude-3-haiku-20240307-v1:0" or
            "amazon.titan-text-express-v1".
            Defaults to utils.config.config.default_model_id if not given
            — this is what makes swapping Claude <-> Titan a one-argument
            change instead of a code rewrite.
        system_prompt: Optional system-level instruction (persona / rules).
            Kept as a separate parameter (not concatenated into `prompt`)
            because Converse has a dedicated `system` field, and models are
            trained to treat system instructions with different priority
            than user turns — merging them would silently degrade quality.
        temperature: Controls randomness of token selection (0.0 - 1.0).
            LOW (e.g. 0.1) -> more deterministic, focused, repeatable
                output. Good for factual Q&A, extraction, code.
            HIGH (e.g. 1.0) -> more varied, creative, sometimes less
                coherent output. Good for brainstorming, creative writing.
        top_p: Nucleus sampling threshold (0.0 - 1.0). The model only
            samples from the smallest set of tokens whose cumulative
            probability >= top_p.
            LOW top_p -> restricts choices to only the most-likely tokens
                (narrower, safer vocabulary).
            HIGH top_p (near 1.0) -> allows sampling from a much larger
                pool of plausible tokens (more lexical variety).
            NOTE: temperature and top_p both control randomness but via
            different mechanisms — this project asks you to vary them
            independently specifically to observe that they are NOT
            redundant with each other.
        max_tokens: Hard cap on generated response length (cost + latency
            control — without this, a model can ramble indefinitely).

    Returns:
        A dict with the fields the notebook/tests actually need:
        {
            "text": "<the model's reply as plain text>",
            "stop_reason": "end_turn" | "max_tokens" | ...,
            "input_tokens": int,
            "output_tokens": int,
        }

    Raises:
        botocore.exceptions.ClientError: e.g. AccessDeniedException if the
        IAM identity hasn't been granted access to the requested model in
        the Bedrock console ("Model access" page) — a very common first-run
        error that's worth calling out explicitly (see README troubleshooting).
    """
    from utils.config import config  # local import keeps default resolution lazy

    resolved_model_id = model_id or config.default_model_id
    client = get_bedrock_runtime_client()

    # Converse expects `messages` as a list of turns, each with a `role`
    # and `content` list of typed blocks. We only ever send one user turn
    # here (this is a "hello world" script, not a multi-turn chat loop),
    # but the schema is inherently multi-turn-ready.
    messages = [{"role": "user", "content": [{"text": prompt}]}]

    # inferenceConfig groups the generation-control knobs (temperature,
    # topP, maxTokens) into a single sub-object — this is exactly the part
    # of the Converse request task #3 asks us to experiment with.
    request_kwargs = {
        "modelId": resolved_model_id,
        "messages": messages,
        "inferenceConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens,
        },
    }

    # system is a top-level, optional field — only included if provided,
    # since Converse rejects an empty system list for some model families.
    if system_prompt:
        request_kwargs["system"] = [{"text": system_prompt}]

    response = client.converse(**request_kwargs)

    # Response shape (same for every model family, that's the whole point
    # of Converse): output.message.content is a list of blocks; for plain
    # text generation there's exactly one block with a "text" key.
    output_text = response["output"]["message"]["content"][0]["text"]

    usage = response.get("usage", {})

    return {
        "text": output_text,
        "stop_reason": response.get("stopReason"),
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
    }


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.converse`
    result = send_prompt(prompt="In one sentence, what is Amazon Bedrock?")
    print("Response:", result["text"])
    print("Stop reason:", result["stop_reason"])
    print(
        f"Tokens — in: {result['input_tokens']}, out: {result['output_tokens']}")
