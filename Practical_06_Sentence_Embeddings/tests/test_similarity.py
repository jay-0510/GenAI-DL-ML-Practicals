"""
tests/test_similarity.py
===========================
Unit tests for src/similarity.py — uses small, hand-built embedding
arrays (not the fake model fixture), since these functions operate on
embeddings directly and don't need a model at all.

What we verify:
    1. compute_similarity_matrix returns 1.0 on the diagonal (every
       sentence is identical to itself) and is symmetric.
    2. find_best_match picks the correct sentence/score/index for an
       unambiguous case.
    3. top_k_matches returns results in descending similarity order.
"""

import numpy as np
import pytest

from src.similarity import compute_similarity_matrix, find_best_match, top_k_matches


@pytest.fixture
def toy_embeddings():
    # 3 orthonormal vectors -> similarity to self = 1.0, to others = 0.0,
    # easy to reason about by hand.
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_similarity_matrix_diagonal_is_one(toy_embeddings):
    matrix = compute_similarity_matrix(toy_embeddings)
    assert np.allclose(np.diag(matrix), 1.0)


def test_similarity_matrix_is_symmetric(toy_embeddings):
    matrix = compute_similarity_matrix(toy_embeddings)
    assert np.allclose(matrix, matrix.T)


def test_find_best_match_picks_closest_sentence(toy_embeddings):
    sentences = ["about cats", "about dogs", "about birds"]
    query_embedding = np.array([0.0, 1.0, 0.0])  # exactly matches "about dogs"

    best_sentence, score, index = find_best_match(query_embedding, toy_embeddings, sentences)

    assert best_sentence == "about dogs"
    assert index == 1
    assert score == pytest.approx(1.0)


def test_top_k_matches_returns_descending_order():
    sentences = ["a", "b", "c"]
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    query = np.array([1.0, 0.0])

    results = top_k_matches(query, embeddings, sentences, k=2)

    assert [r[0] for r in results] == ["a", "b"]
    assert results[0][1] >= results[1][1]
