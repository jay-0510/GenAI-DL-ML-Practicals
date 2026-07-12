"""
utils package
=============
Shared configuration and the single Bedrock client factory used by both
`src/variants.py` and `src/roles.py`. See config.py for why the client
factory lives here rather than in its own file for this particular practical.
"""

from .config import config, BedrockConfig, get_bedrock_runtime_client

__all__ = ["config", "BedrockConfig", "get_bedrock_runtime_client"]
