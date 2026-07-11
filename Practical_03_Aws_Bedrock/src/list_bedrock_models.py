"""
src/list_bedrock_models.py
============================

WHAT THIS FILE DOES
--------------------
Satisfies practical task #1: "Write a script to list all available Bedrock
foundation models." It calls the Bedrock control-plane API
(ListFoundationModels) and returns/prints them in a readable form.

WHY THIS IS ITS OWN FILE (not merged into bedrock_client.py or converse.py)
-----------------------------------------------------------------------------
bedrock_client.py only builds connections; it deliberately contains zero
business logic. converse.py deliberately only knows about *running*
inference. "Listing models" is a distinct concern from both — it's a
discovery/inventory operation, not a client-factory or an inference call —
so it gets its own module. This keeps each file testable in isolation:
you can unit-test "did we parse the ListFoundationModels response
correctly?" without touching anything related to Converse.
"""

from typing import Optional

from src.bedrock_client import get_bedrock_client


def list_foundation_models(provider_filter: Optional[str] = None) -> list[dict]:
    """
    Fetches all Bedrock foundation models available in the configured region,
    optionally filtered by provider (e.g. "Anthropic", "Amazon", "Meta").

    Why filter by provider here instead of just always returning everything?
        In a real account there can be 50+ models across many providers.
        Filtering server-adjacent (right after the API call) keeps the
        function directly useful for "just show me the Claude models" type
        calls, which is the common case for this project (we only care
        about Claude/Titan for the hello-world task).

    Args:
        provider_filter: Case-insensitive substring match against
            'providerName' (e.g. "anthropic"). None returns all providers.

    Returns:
        A list of dicts, each describing one model, e.g.:
        {
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "providerName": "Anthropic",
            "modelName": "Claude 3 Haiku",
            "inputModalities": ["TEXT"],
            "outputModalities": ["TEXT"],
        }

    Raises:
        botocore.exceptions.ClientError: if the IAM identity calling this
        lacks bedrock:ListFoundationModels permission, or the region has no
        Bedrock endpoint.
    """
    client = get_bedrock_client()

    # summaries: response key returned by ListFoundationModels containing
    # one entry per model available to this account in this region.
    response = client.list_foundation_models()
    models = response.get("modelSummaries", [])

    if provider_filter:
        needle = provider_filter.lower()
        models = [m for m in models if needle in m.get("providerName", "").lower()]

    # Trim down to the fields that actually matter for this project so
    # callers (and the notebook) aren't drowning in metadata like
    # 'customizationsSupported' that we never use.
    return [
        {
            "modelId": m.get("modelId"),
            "providerName": m.get("providerName"),
            "modelName": m.get("modelName"),
            "inputModalities": m.get("inputModalities"),
            "outputModalities": m.get("outputModalities"),
        }
        for m in models
    ]


def print_models_table(models: list[dict]) -> None:
    """
    Pretty-prints a list of model dicts as an aligned table.

    Kept separate from list_foundation_models() on purpose: that function
    returns *data* (so tests and the notebook can consume it programmatically),
    while this function is purely a *presentation* concern (so it's the only
    thing that changes if we later want JSON output, CSV output, etc.).
    """
    if not models:
        print("No models found (check region / provider filter / IAM permissions).")
        return

    header = f"{'MODEL ID':<45} {'PROVIDER':<12} {'MODEL NAME':<25}"
    print(header)
    print("-" * len(header))
    for m in models:
        print(f"{m['modelId']:<45} {m['providerName']:<12} {m['modelName']:<25}")


if __name__ == "__main__":
    # Allows running this file directly: `python -m src.list_bedrock_models`
    # Useful for the CLI part of the task, separate from notebook usage.
    all_models = list_foundation_models()
    print(f"Found {len(all_models)} total foundation models.\n")
    print_models_table(all_models)

    print("\n--- Anthropic models only ---")
    print_models_table(list_foundation_models(provider_filter="anthropic"))
