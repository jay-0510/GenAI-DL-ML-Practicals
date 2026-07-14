"""
utils package
=============
Shared settings for the embedding model, t-SNE reproducibility, and
output paths. See config.py for why this practical needs no
AWS/dotenv setup, unlike the Bedrock-based practicals before it.
"""

from .config import config, EmbeddingConfig

__all__ = ["config", "EmbeddingConfig"]
