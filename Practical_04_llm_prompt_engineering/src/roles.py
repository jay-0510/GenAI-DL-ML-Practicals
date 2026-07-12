"""
src/roles.py
=============

WHAT THIS FILE DOES
--------------------
Covers practical task 2: set the model's SYSTEM message to different
personas — "senior data analyst" vs "creative writer" — send it the SAME
user prompt, and compare tone/quality differences.

WHY THIS IS A SEPARATE FILE FROM variants.py
-----------------------------------------------
Task 1 (variants.py) varies the USER prompt's structure (zero-shot vs
few-shot vs CoT) while the system message stays empty/neutral. Task 2
(this file) does the opposite: the user prompt stays FIXED and only the
SYSTEM message changes. Keeping these as separate experiments in separate
files makes each one a clean, single-variable comparison — mixing both
prompt-technique AND persona changes into one file would make it unclear
which variable caused an observed difference in output.

This file reuses `call_bedrock()` from `variants.py` for the actual API
call (see variants.py's docstring for why that function is the single
place the Converse API gets invoked) — it only owns the persona
definitions and the comparison logic.
"""

from src.variants import call_bedrock

# Two contrasting system personas. Deliberately opposite in register (dry
# and analytical vs expressive and narrative) so tone differences in the
# output are unambiguous rather than subtle.
SYSTEM_PROMPTS = {
    "senior_data_analyst": (
        "You are a senior data analyst. Respond precisely and objectively. "
        "Use structured, factual language, quantify things where possible, "
        "avoid embellishment, and get to the point quickly."
    ),
    "creative_writer": (
        "You are an imaginative creative writer. Respond with vivid, "
        "expressive language, use metaphor and narrative flair where it "
        "fits, and prioritize engaging, colorful phrasing over brevity."
    ),
}


def get_system_prompt(role: str) -> str:
    """
    Looks up the system message text for a given persona key.

    Raises:
        ValueError: on an unknown role, so a typo'd role name fails loudly
        at call time instead of silently sending an empty system message
        (which would make the "role had no effect" result misleading).
    """
    if role not in SYSTEM_PROMPTS:
        raise ValueError(
            f"Unknown role '{role}'. Expected one of {list(SYSTEM_PROMPTS)}."
        )
    return SYSTEM_PROMPTS[role]


def run_with_role(prompt: str, role: str, **call_kwargs) -> dict:
    """
    Sends `prompt` to Bedrock with the given persona set as the system
    message.

    Args:
        prompt: the (fixed) user prompt — kept identical across roles so
            any output difference is attributable ONLY to the system
            message, not to prompt wording.
        role: one of the keys in SYSTEM_PROMPTS.
        **call_kwargs: forwarded to call_bedrock (temperature, top_p, etc.)
            — left at config defaults by default so this experiment isn't
            accidentally confounded by ALSO varying generation settings
            between the two persona calls.
    """
    result = call_bedrock(
        user_prompt=prompt, system_prompt=get_system_prompt(role), **call_kwargs
    )
    result["role"] = role
    return result


def compare_roles(prompt: str, roles: list[str] | None = None, **call_kwargs) -> dict:
    """
    Runs the SAME prompt under each requested persona and returns results
    keyed by role name, for side-by-side tone/quality comparison.

    Args:
        prompt: fixed user prompt sent to every role.
        roles: which personas to compare; defaults to ALL personas in
            SYSTEM_PROMPTS (currently senior_data_analyst and
            creative_writer) so adding a third persona later needs no
            change here.
    """
    roles = roles or list(SYSTEM_PROMPTS)
    return {role: run_with_role(prompt, role, **call_kwargs) for role in roles}


if __name__ == "__main__":
    # Manual smoke test: `python -m src.roles`
    sample_prompt = "Explain why employee turnover is a problem worth solving."
    for role_name, res in compare_roles(sample_prompt).items():
        print(f"[{role_name}] ->\n{res['text']}\n")
