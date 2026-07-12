"""
utils/config.py
================

WHAT THIS FILE HOLDS, AND WHY IT'S BIGGER THAN JUST "SETTINGS" HERE
----------------------------------------------------------------------
In practical_03 (AWS Bedrock Hello World), client construction lived in its
own `src/bedrock_client.py` because that project needed TWO different
Bedrock clients (control-plane "bedrock" for listing models, and data-plane
"bedrock-runtime" for inference).

This practical (Prompt Engineering) never lists models — every experiment
here is "send a prompt, read the response," so only the ONE runtime client
is ever needed. Giving that single function its own file would be an empty
module for the sake of empty modules, so it lives here instead, right next
to the settings it depends on. If a future practical in this repo needs the
control-plane client too, split it back out then — one clear function is
cheap to move.

Everything else about the reasoning is the same as before:
    - one typed, immutable settings object (no scattered magic strings)
    - values sourced from environment variables via python-dotenv, never
      hardcoded, so nothing AWS-specific ends up in git history
    - the client is cached (@lru_cache) so we only pay boto3's client
      construction cost once per process
"""

import os
from dataclasses import dataclass
from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class BedrockConfig:
    """
    Immutable settings shared by every module in this project.

    frozen=True -> no module can accidentally mutate shared config
    mid-notebook-run, which would otherwise be a very confusing bug when
    running the same cell twice with "should be" identical settings.
    """

    region_name: str = os.getenv("AWS_REGION", "us-east-1")

    # Default model for all prompt-engineering experiments in this practical.
    default_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0"
    )

    default_max_tokens: int = int(os.getenv("BEDROCK_MAX_TOKENS", "512"))
    default_temperature: float = float(os.getenv("BEDROCK_TEMPERATURE", "0.3"))
    default_top_p: float = float(os.getenv("BEDROCK_TOP_P", "0.9"))
    max_retry_attempts: int = int(os.getenv("BEDROCK_MAX_RETRIES", "3"))


config = BedrockConfig()


@lru_cache(maxsize=1)
def get_bedrock_runtime_client():
    """
    Returns a cached boto3 client for the Bedrock DATA PLANE
    ("bedrock-runtime" service) — the only Bedrock service this practical
    needs, since every experiment here is a Converse call, never a
    "list models" control-plane call.

    Why @lru_cache?
        boto3 clients are safe to reuse and non-trivial to construct
        (they parse service definitions on first creation). Every function
        in src/variants.py and src/roles.py calls this instead of building
        its own client, so we build it once and share it.

    Raises:
        RuntimeError: friendly message if AWS credentials aren't configured,
        instead of letting botocore's raw exception surface.
    """
    try:
        return boto3.client(
            "bedrock-runtime",
            config=Config(
                region_name=config.region_name,
                retries={"max_attempts": config.max_retry_attempts, "mode": "adaptive"},
            ),
        )
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise RuntimeError(
            "AWS credentials not found. Run `aws configure` (or set "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars) before "
            "using this client."
        ) from exc
