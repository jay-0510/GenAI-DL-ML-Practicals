"""
utils package (faiss_search)
=============================
Settings for the local FAISS half of this practical -- no AWS, no
secrets, just the embedding model name and default search size.
"""

from .config import config, FaissConfig

__all__ = ["config", "FaissConfig"]
