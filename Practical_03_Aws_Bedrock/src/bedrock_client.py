"""
src/bedrock_client.py
======================

WHAT THIS FILE DOES
--------------------
AWS Bedrock is actually TWO separate services under one name:

  1. "bedrock"          -> the CONTROL PLANE. Used to list/describe available
                             foundation models, check model access, etc.
                             (used by list_bedrock_models.py)

  2. "bedrock-runtime"   -> the DATA PLANE. Used to actually run inference
                             against a model (Converse, InvokeModel).
                             (used by converse.py)

This module is the ONLY place in the whole project that calls
`boto3.client(...)`. Every other module asks THIS file for a client instead
of constructing one itself.

WHY CENTRALIZE CLIENT CREATION?
--------------------------------
1. Consistency: region, retry policy, and timeouts are configured once
   (via utils/config.py) and apply identically to every AWS call in the
   project — no risk of one file quietly using a different region.
2. Testability: unit tests only need to mock two functions
   (get_bedrock_client / get_bedrock_runtime_client) to fake ALL AWS
   interaction in the project, instead of patching boto3.client() in five
   different files.
3. Fail-fast error handling: if AWS credentials are missing/invalid, we
   want ONE clear error message the first time a client is requested, not
   five different cryptic botocore tracebacks scattered across modules.
"""

from functools import lru_cache

import boto3
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

from utils.config import config


def _build_retry_config() -> Config:
    """
    Builds the shared botocore Config object (retries, region).

    Why 'adaptive' retry mode specifically?
        Bedrock throttles aggressively on shared/trial accounts
        (ThrottlingException). 'adaptive' mode uses a client-side rate
        limiter that backs off dynamically based on observed throttling,
        which recovers faster than the older 'standard' fixed-backoff mode.
    """
    return Config(
        region_name=config.region_name,
        retries={
            "max_attempts": config.max_retry_attempts,
            "mode": "adaptive",
        },
    )


@lru_cache(maxsize=1)
def get_bedrock_client():
    """
    Returns a cached boto3 client for the Bedrock CONTROL PLANE
    (service name: "bedrock").

    Use this client for: ListFoundationModels, GetFoundationModel,
    ListModelAccess — anything that is about *metadata*, not inference.

    Why @lru_cache?
        boto3 clients are thread-safe and reasonably expensive to build
        (they parse service models on first creation). Since our config
        never changes mid-run, we build the client once and reuse it
        instead of re-constructing it on every function call.

    Raises:
        RuntimeError: if AWS credentials aren't configured, with a message
        that tells the user exactly how to fix it (aws configure), rather
        than surfacing boto3's raw, less-friendly exception.
    """
    try:
        return boto3.client("bedrock", config=_build_retry_config())
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise RuntimeError(
            "AWS credentials not found. Run `aws configure` (or set "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars) before "
            "using this client."
        ) from exc


@lru_cache(maxsize=1)
def get_bedrock_runtime_client():
    """
    Returns a cached boto3 client for the Bedrock DATA PLANE
    (service name: "bedrock-runtime").

    Use this client for: Converse, ConverseStream, InvokeModel — anything
    that actually sends a prompt to a model and gets generated output back.

    Kept as a SEPARATE function (not a parameter on get_bedrock_client)
    because "bedrock" and "bedrock-runtime" are genuinely different AWS
    service endpoints with different IAM permissions — conflating them
    would hide that distinction from callers and from IAM policy reviews.
    """
    try:
        return boto3.client("bedrock-runtime", config=_build_retry_config())
    except (NoCredentialsError, PartialCredentialsError) as exc:
        raise RuntimeError(
            "AWS credentials not found. Run `aws configure` (or set "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars) before "
            "using this client."
        ) from exc

@lru_cache(maxsize=1)
def get_bedrock_client():
    import boto3

    session = boto3.Session()

    print("Region:", session.region_name)
    print("Credentials:", session.get_credentials())
    print("Available profiles:", session.available_profiles)

    return session.client(
        "bedrock",
        region_name=config.region_name
    )