"""
faiss_search/src/embedder.py
===============================

WHAT THIS FILE DOES
--------------------
Loads the SBERT model and encodes sentences into embedding vectors that
FAISS can index.

WHY SBERT?
------------
Plain BERT wasn't designed to produce sentence-level embeddings that are
directly comparable by distance/similarity -- averaging its token
embeddings gives poor nearest-neighbor results in practice. SBERT is
fine-tuned specifically so that semantically similar sentences end up
close together in embedding space, which is the exact property FAISS's
nearest-neighbor search depends on.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import config


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Loads (and caches) the SBERT model -- see the matching docstring
    in Practical 6's embedder.py for the full reasoning on why this is
    cached rather than reloaded per call."""
    return SentenceTransformer(config.model_name)


def encode_sentences(sentences: List[str]) -> np.ndarray:
    """
    Encodes sentences into a 2D float32 array, shape
    (len(sentences), embedding_dim).

    normalize_embeddings=True AND the explicit float32 cast both matter
    for FAISS specifically: normalized vectors let us use FAISS's inner
    product index (IndexFlatIP) as an exact stand-in for cosine
    similarity (see faiss_index.py's docstring), and FAISS requires
    float32 arrays -- sentence-transformers can return float64 depending
    on the backend, which FAISS will silently reject or mishandle if not
    cast explicitly.
    """
    model = load_model()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


if __name__ == "__main__":
    # Manual smoke test: `python -m src.embedder`
    from src.data import get_sentences

    sentences = get_sentences()
    embeddings = encode_sentences(sentences)
    print(f"Encoded {len(sentences)} sentences into shape {embeddings.shape}, dtype {embeddings.dtype}")
