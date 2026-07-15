"""
src/embedder.py
==================

WHAT THIS FILE DOES
--------------------
Loads the SBERT model and encodes sentences into embedding vectors.

WHY SBERT INSTEAD OF PLAIN BERT?
-------------------------------------
Plain BERT wasn't designed to produce sentence-level embeddings that are
directly comparable via cosine similarity — averaging its token
embeddings ("mean-pooling a vanilla BERT") produces surprisingly poor
similarity rankings in practice. SBERT is BERT specifically fine-tuned
(via a siamese/triplet network setup during training) so that
semantically similar sentences end up CLOSE together in embedding space
and dissimilar ones end up FAR apart — exactly the property both cosine
similarity search (similarity.py) and t-SNE clustering (visualize.py)
depend on.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.config import config


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """
    Loads (and caches) the SBERT model.

    Why @lru_cache? Loading a transformer model means reading tens of MB
    of weights off disk (or downloading them once, the first time) —
    slow enough that doing it on every single encode_sentences() call
    would make this practical noticeably sluggish. Caching means the
    model loads exactly once per notebook/process, no matter how many
    times encode_sentences() gets called afterward.
    """
    return SentenceTransformer(config.model_name)


def encode_sentences(sentences: List[str]) -> np.ndarray:
    """
    Encodes a list of sentences into a 2D array of embeddings, shape
    (len(sentences), embedding_dim) — 384-dim for all-MiniLM-L6-v2.

    normalize_embeddings=True makes every output vector unit-length.
    This is what lets similarity.py compute cosine similarity as a plain
    dot product later (the cosine similarity of two unit vectors IS
    their dot product) — a small speed win, but more importantly it
    removes a common source of subtly-wrong similarity scores if that
    normalization step were ever done inconsistently downstream instead.
    """
    model = load_model()
    return model.encode(sentences, normalize_embeddings=True)


if __name__ == "__main__":
    # Manual smoke test: `python -m src.embedder`
    from src.data import get_sentences

    sentences = get_sentences()
    embeddings = encode_sentences(sentences)
    print(f"Encoded {len(sentences)} sentences into shape {embeddings.shape}")
