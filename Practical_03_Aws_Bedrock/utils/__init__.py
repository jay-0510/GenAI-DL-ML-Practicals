"""
utils package
=============
Cross-cutting helpers shared across `src/` and `tests/`. Currently just
`config.py`, but kept as its own package (rather than dumping config.py into
`src/`) so it's obvious this code has no AWS-calling logic of its own — it's
pure configuration/plumbing that both application code and test code depend
on symmetrically.
"""

from .config import config, BedrockConfig

__all__ = ["config", "BedrockConfig"]
