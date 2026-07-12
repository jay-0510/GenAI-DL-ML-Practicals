"""
src/variants.py
=================

WHAT THIS FILE DOES
--------------------
Covers practical task 1: compare zero-shot, few-shot, and chain-of-thought
(CoT) prompting on the SAME classification task, using the Bedrock Converse
API.

THE CLASSIFICATION TASK
-------------------------
We classify an HR helpdesk ticket's free-text body into exactly one of four
categories:

    "Payroll"            - pay, salary, deductions, payslips, reimbursement
    "Leave & Attendance"  - leave balance, WFH requests, attendance regularisation
    "IT Support"          - laptop/VPN/account/access issues
    "Policy & General"    - policy questions, general HR queries

(Chosen because it's a realistic, unambiguous-enough task to grade by eye,
with categories that are easy to confuse — which is exactly what makes
prompt-engineering technique actually matter here.)

WHY THIS FILE ALSO OWNS THE LOW-LEVEL BEDROCK CALL
-----------------------------------------------------
`call_bedrock()` below is the ONE function in this whole project that
actually calls `client.converse(...)`. `src/roles.py` (task 2 — persona
system messages) imports and reuses THIS function rather than making its
own Bedrock call, so there is a single place that builds the Converse
request shape. This avoids two files independently getting subtly
different request formats and keeps mocking simple in tests (patch this
one function, both roles.py and variants.py tests are covered).
"""

from typing import Optional

from utils.config import config, get_bedrock_runtime_client

# The fixed label set every prompt variant must classify into. Defined once
# here so all three prompt builders (and the notebook) reference the exact
# same wording — a mismatch between "IT Support" in one prompt and
# "IT" in another would silently make outputs incomparable.
CATEGORIES = ["Payroll", "Leave & Attendance", "IT Support", "Policy & General"]


# ---------------------------------------------------------------------------
# Low-level Bedrock call (shared by variants.py and roles.py)
# ---------------------------------------------------------------------------
def call_bedrock(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[float] = None,
    model_id: Optional[str] = None,
) -> dict:
    """
    Sends a single-turn prompt to Bedrock via the Converse API and returns
    a small, flat result dict.

    Why Converse (not invoke_model)?
        Converse gives one request/response shape regardless of which
        model family is behind `model_id` — useful here because we might
        want to re-run the exact same experiment against a different model
        later without rewriting any request-building code.

    Args:
        user_prompt: The fully-built prompt text (already containing the
            task instructions / examples / CoT scaffolding — this function
            doesn't know or care which prompting technique built it, which
            is what lets zero-shot/few-shot/CoT all funnel through the
            same call).
        system_prompt: Optional system message. Used by BOTH:
            - task 1's prompt-variant experiments (usually left as None
              here, since the technique being tested is in the user turn),
            - task 2's persona experiments in roles.py (where THIS is the
              actual variable under test).
        temperature / top_p / max_tokens / model_id: fall back to
            utils.config.config defaults when not given, so callers only
            need to override what they're actually experimenting with.

    Returns:
        {"text": str, "stop_reason": str, "input_tokens": int, "output_tokens": int}
    """
    client = get_bedrock_runtime_client()

    request_kwargs = {
        "modelId": model_id or config.default_model_id,
        "messages": [{"role": "user", "content": [{"text": user_prompt}]}],
        "inferenceConfig": {
            "temperature": temperature if temperature is not None else config.default_temperature,
            "topP": top_p if top_p is not None else config.default_top_p,
            "maxTokens": int(max_tokens) if max_tokens is not None else config.default_max_tokens,
        },
    }
    if system_prompt:
        request_kwargs["system"] = [{"text": system_prompt}]

    response = client.converse(**request_kwargs)
    output_text = response["output"]["message"]["content"][0]["text"]
    usage = response.get("usage", {})

    return {
        "text": output_text,
        "stop_reason": response.get("stopReason"),
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
    }


# ---------------------------------------------------------------------------
# Prompt builders — one per technique, same task, same output categories
# ---------------------------------------------------------------------------
def build_zero_shot_prompt(ticket_text: str) -> str:
    """
    Zero-shot: give the model ONLY the task instructions and category list
    — no worked examples, no reasoning scaffold. This is the cheapest
    technique (shortest prompt, fewest tokens) and the baseline every other
    technique should be measured against.

    Best used when: the task is simple/unambiguous enough that the model's
    pretraining alone covers it, or when prompt length/cost is tightly
    constrained.
    """
    categories = ", ".join(CATEGORIES)
    return (
        "Classify the following HR helpdesk ticket into exactly one of these "
        f"categories: {categories}.\n"
        "Respond with ONLY the category name, nothing else.\n\n"
        f"Ticket: \"{ticket_text}\"\n"
        "Category:"
    )


def build_few_shot_prompt(ticket_text: str) -> str:
    """
    Few-shot: show the model 2-3 worked (input -> correct label) examples
    before asking it to classify the real ticket. This "primes" the model
    with the exact output format and decision boundary we want, which
    tends to reduce ambiguous/borderline misclassifications that zero-shot
    is prone to (e.g. a ticket that mentions both "salary" and "leave").

    Best used when: categories are easy to confuse, or you need the output
    in a very specific format that zero-shot doesn't reliably produce.
    """
    categories = ", ".join(CATEGORIES)
    examples = (
        'Ticket: "My salary credited this month is less than usual, why?"\n'
        "Category: Payroll\n\n"
        'Ticket: "I want to apply for 3 days of casual leave next week."\n'
        "Category: Leave & Attendance\n\n"
        'Ticket: "My laptop won\'t connect to the office VPN."\n'
        "Category: IT Support\n\n"
    )
    return (
        "Classify HR helpdesk tickets into exactly one of these categories: "
        f"{categories}.\n"
        "Respond with ONLY the category name, nothing else.\n\n"
        "Here are some examples:\n\n"
        f"{examples}"
        f'Now classify this ticket:\nTicket: "{ticket_text}"\n'
        "Category:"
    )


def build_cot_prompt(ticket_text: str) -> str:
    """
    Chain-of-thought (CoT): explicitly ask the model to reason step by step
    BEFORE giving the final label, and to put the final label on its own
    clearly-marked line. This tends to help most on genuinely ambiguous
    tickets (mentions of multiple topics) because the model is forced to
    weigh the evidence instead of pattern-matching to the first keyword it
    notices.

    Trade-off worth documenting: CoT uses more output tokens (higher cost/
    latency) and its final answer is slightly harder to parse programmatically
    since you must extract the label from free text rather than reading the
    whole response as the label.

    Best used when: categories overlap semantically, or you want the
    reasoning itself for auditing/debugging why a ticket was routed a
    certain way (useful in a real HR system for QA of the routing logic).
    """
    categories = ", ".join(CATEGORIES)
    return (
        "Classify the following HR helpdesk ticket into exactly one of these "
        f"categories: {categories}.\n\n"
        f'Ticket: "{ticket_text}"\n\n'
        "Think step by step: identify the key topic(s) mentioned in the "
        "ticket, weigh which category they best match, then give your "
        "final answer.\n\n"
        "Format your response as:\n"
        "Reasoning: <your step-by-step reasoning>\n"
        "Category: <one of the category names, exactly as listed above>"
    )


_PROMPT_BUILDERS = {
    "zero_shot": build_zero_shot_prompt,
    "few_shot": build_few_shot_prompt,
    "cot": build_cot_prompt,
}


def classify_ticket(ticket_text: str, variant: str = "zero_shot", **call_kwargs) -> dict:
    """
    Builds the requested prompt variant and sends it to Bedrock.

    Args:
        ticket_text: raw HR ticket text to classify.
        variant: one of "zero_shot", "few_shot", "cot".
        **call_kwargs: passed through to call_bedrock (temperature, top_p,
            max_tokens, model_id) so the notebook can override generation
            settings per call without this function needing to know about them.

    Raises:
        ValueError: if `variant` isn't one of the three supported techniques
            — fails loudly rather than silently defaulting, since silently
            falling back to zero-shot would invalidate a comparison.
    """
    if variant not in _PROMPT_BUILDERS:
        raise ValueError(
            f"Unknown variant '{variant}'. Expected one of {list(_PROMPT_BUILDERS)}."
        )
    prompt = _PROMPT_BUILDERS[variant](ticket_text)
    result = call_bedrock(user_prompt=prompt, **call_kwargs)
    result["variant"] = variant
    return result


def compare_variants(ticket_text: str, **call_kwargs) -> dict:
    """
    Runs all three prompting techniques against the SAME ticket text and
    returns their results keyed by variant name, so they can be printed or
    tabulated side by side.

    Kept as a thin loop over classify_ticket() rather than its own
    duplicated logic — if a 4th technique is ever added to
    _PROMPT_BUILDERS, this function needs zero changes.
    """
    return {
        variant: classify_ticket(ticket_text, variant=variant, **call_kwargs)
        for variant in _PROMPT_BUILDERS
    }


if __name__ == "__main__":
    # Manual smoke test: `python -m src.variants`
    sample_ticket = "I haven't received my reimbursement for last month's travel expenses."
    for variant_name, res in compare_variants(sample_ticket).items():
        print(f"[{variant_name}] -> {res['text']}\n")
