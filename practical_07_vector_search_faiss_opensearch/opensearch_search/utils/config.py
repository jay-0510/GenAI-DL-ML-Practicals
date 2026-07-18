"""
opensearch_search/utils/config.py
====================================

WHAT THIS FILE HOLDS
----------------------
Settings for the AWS half of the practical: region, collection/index
names, the embedding model, and vector dimension.

WHY SBERT LOCALLY, NOT A BEDROCK EMBEDDING MODEL (e.g. Titan Embeddings)?
-------------------------------------------------------------------------------
Three reasons:
  1. Consistency -- using the SAME model as faiss_search/ means both
     halves of this practical embed text into the SAME 384-dimensional
     space, so "does OpenSearch's semantic search behave like FAISS's?"
     is a fair comparison, not confounded by two different embedding
     models with different notions of "similar."
  2. Scope -- this practical is about OpenSearch itself (creating a
     vector-capable index, running keyword/semantic/hybrid queries), not
     about Bedrock. Pulling in a second AWS service (Bedrock) just to
     generate vectors would add an extra piece of AWS setup unrelated to
     what's actually being taught here.
  3. Cost/setup -- no extra Bedrock model access request needed; the only
     AWS service this sub-project touches is OpenSearch Serverless itself.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class OpenSearchConfig:
    region_name: str = os.getenv("AWS_REGION", "us-east-1")

    # Same SBERT model as faiss_search/ -- see module docstring for why.
    model_name: str = os.getenv("SBERT_MODEL_NAME", "all-MiniLM-L6-v2")
    embedding_dimension: int = 384  # must match all-MiniLM-L6-v2's output size

    # Names for the AWS-side resources this sub-project creates.
    collection_name: str = os.getenv("OPENSEARCH_COLLECTION_NAME", "practical7-vector-search")
    index_name: str = os.getenv("OPENSEARCH_INDEX_NAME", "documents")

    # How many results each query type returns by default.
    default_top_k: int = int(os.getenv("OPENSEARCH_TOP_K", "5"))

    # How many results keyword_search and semantic_search each fetch
    # BEFORE fusion, when used inside hybrid_search -- wider than
    # default_top_k because rank fusion needs enough candidates from each
    # side to meaningfully re-rank, not just the final top few.
    hybrid_candidate_pool: int = int(os.getenv("OPENSEARCH_HYBRID_POOL", "20"))


config = OpenSearchConfig()
