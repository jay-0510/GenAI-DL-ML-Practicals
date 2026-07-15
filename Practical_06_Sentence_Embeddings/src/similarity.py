"""
src/similarity.py
====================

WHAT THIS FILE DOES
--------------------
Computes pairwise cosine similarity across all sentence embeddings, and
finds the best-matching sentence(s) for a new query.

WHY COSINE SIMILARITY, NOT EUCLIDEAN DISTANCE?
----------------------------------------------------
Cosine similarity measures the ANGLE between two vectors, ignoring their
magnitude — appropriate here because SBERT embeddings encode MEANING in
direction, not length. Euclidean distance is also sensitive to how
"long" an embedding vector happens to be, which isn't semantically
meaningful for these embeddings and would distort rankings. Since
embedder.py already normalizes every vector to unit length, cosine
similarity here reduces to a plain dot product — see encode_sentences()'s
docstring for why that shortcut is valid.
"""

from typing import List, Tuple

import numpy as np


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Returns an (n, n) matrix where entry [i, j] is the cosine similarity
    between sentence i and sentence j. Values range from -1 (opposite
    meaning) to 1 (identical meaning); the diagonal is always 1.0 (every
    sentence is identical to itself).

    Implemented as `embeddings @ embeddings.T` (a single matrix multiply)
    rather than a nested loop calling a pairwise function n*n times —
    numpy's vectorized multiply computes every pair in one call, which is
    both far faster and far shorter than looping in Python.
    """
    return embeddings @ embeddings.T


def find_best_match(
    query_embedding: np.ndarray, sentence_embeddings: np.ndarray, sentences: List[str]
) -> Tuple[str, float, int]:
    """
    Finds which of `sentences` is most similar to a single query embedding.

    Returns:
        (best_sentence, similarity_score, index) — the index is included
        because callers (e.g. the notebook) often also want that
        sentence's topic label, which lives in a separate parallel list
        (src.data.get_topics()) at the same position.
    """
    scores = sentence_embeddings @ query_embedding
    best_index = int(np.argmax(scores))
    return sentences[best_index], float(scores[best_index]), best_index


def top_k_matches(
    query_embedding: np.ndarray,
    sentence_embeddings: np.ndarray,
    sentences: List[str],
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Returns the top-k most similar sentences to a query, best first.

    Kept as its own function rather than a "k=1" special case of
    find_best_match() — "give me THE best match" (a single
    sentence/score/index tuple) and "give me the top few, ranked" (a
    list) are used in different parts of the notebook and read more
    clearly as two small, differently-shaped functions than one function
    whose return type changes based on an argument.
    """
    scores = sentence_embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:k]
    return [(sentences[i], float(scores[i])) for i in top_indices]
