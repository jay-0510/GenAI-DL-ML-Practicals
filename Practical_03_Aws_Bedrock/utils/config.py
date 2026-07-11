"""
utils/config.py
================

WHY THIS FILE EXISTS
---------------------
Every other module in this project (bedrock_client.py, converse.py,
list_bedrock_models.py, and both test files) needs the same handful of
settings: which AWS region to talk to, which model to default to, how many
tokens to generate, etc.

Instead of scattering `region_name = "us-east-1"` and magic numbers across
five files, we define ONE typed config object here. Benefits:

1. Single source of truth -> change the region/model once, it propagates
   everywhere.
2. Values come from environment variables (via `.env` + python-dotenv),
   never hardcoded -> no AWS account details leak into git history.
3. Easy to override in tests: monkeypatch the env var, the config picks it
   up automatically instead of needing to patch every file individually.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the process
# environment. In CI / production, these would instead be real environment
# variables injected by the platform, so this call is a no-op there.
load_dotenv()


@dataclass(frozen=True)
class BedrockConfig:
    """
    Immutable (frozen=True) container for all Bedrock-related settings.

    Why a dataclass instead of a plain dict?
        - Typos become AttributeErrors at development time (config.regoin
          fails immediately) instead of silently returning None like a
          dict lookup would.
        - `frozen=True` prevents any module from accidentally mutating
          shared config mid-run, which would be a hard-to-debug bug in a
          multi-module project like this one.
    """

    # AWS region. Bedrock model availability differs by region, so this is
    # the single most important setting in the whole project.
    region_name: str = (
    os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "us-east-1"
)
    # Default foundation model to use if a caller doesn't specify one.
    # Amazon Nova Micro is used as the default here because it's
    # fast and cheap, which is ideal for a "hello world" experiment.
    default_model_id: str = os.getenv(
        "BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0"
    )

    # Generation parameters (overridable per-call in converse.py, but these
    # act as safe defaults so every function doesn't need every argument).
    default_max_tokens: int = int(os.getenv("BEDROCK_MAX_TOKENS", "512"))
    default_temperature: float = float(os.getenv("BEDROCK_TEMPERATURE", "0.5"))
    default_top_p: float = float(os.getenv("BEDROCK_TOP_P", "0.9"))

    # boto3 retry configuration. Bedrock enforces per-account throttling
    # (ThrottlingException) especially on shared/free-tier accounts, so we
    # bake in retries here rather than having every caller handle it.
    max_retry_attempts: int = int(os.getenv("BEDROCK_MAX_RETRIES", "3"))


# A single module-level instance. Importing `config` from this module gives
# every file the exact same settings without needing to re-instantiate the
# dataclass everywhere (cheap, but keeps things consistent and explicit).
config = BedrockConfig()
