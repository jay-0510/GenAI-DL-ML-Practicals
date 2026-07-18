"""
opensearch_search/src/embedder.py
====================================

WHAT THIS FILE DOES
--------------------
Loads the SBERT model and encodes text into embedding vectors that get
stored in OpenSearch's knn_vector field.

Same model and same reasoning as faiss_search/src/embedder.py (see that
file's docstring) -- kept as its own copy here, rather than importing
across the faiss_search/opensearch_search boundary, so each sub-project
stays fully self-contained and independently runnable. See
utils/config.py's docstring for why SBERT (not a Bedrock embedding
model) is used on the AWS side of this practical too.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import config


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Loads (and caches) the SBERT model once per process."""
    return SentenceTransformer(config.model_name)


def encode_texts(texts: List[str]) -> np.ndarray:
    """
    Encodes text into a 2D float32 array, shape (len(texts), embedding_dimension).

    normalize_embeddings=True matters here for the SAME reason as in
    faiss_search/: OpenSearch's knn_vector field is configured with
    space_type="innerproduct" (see indexer.py's docstring), which is only
    equivalent to cosine similarity when the stored vectors are unit
    length -- so normalization has to happen here, at encode time, not
    left to OpenSearch to handle.
    """
    model = load_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    # Manual smoke test: `python -m src.embedder`
    from src.data import get_documents

    texts = [doc["text"] for doc in get_documents()]
    embeddings = encode_texts(texts)
    print(f"Encoded {len(texts)} documents into shape {embeddings.shape}")
