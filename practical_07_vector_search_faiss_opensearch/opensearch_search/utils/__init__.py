"""
utils package (opensearch_search)
====================================
AWS region, OpenSearch Serverless collection/index names, and the
embedding model/dimension shared with faiss_search/ for a fair
comparison. See config.py for why SBERT (not a Bedrock embedding model)
is used here.
"""

from .config import config, OpenSearchConfig

__all__ = ["config", "OpenSearchConfig"]
