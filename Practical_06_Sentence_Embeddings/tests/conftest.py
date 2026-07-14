"""
tests/conftest.py
====================

WHY THIS FILE EXISTS
----------------------
The biggest cost in testing embedder.py (and anything downstream of it)
naively would be actually loading the real SBERT model — slow, downloads
weights on first use, needs internet. `fake_sbert_model` sidesteps that
entirely with a lightweight fake that returns deterministic, hand-picked
vectors instead of real embeddings — these tests check OUR code's logic
(shapes, normalization, wiring), not whether the real model produces
semantically good embeddings, so a fake is not just faster but actually
the RIGHT tool for what's being tested.

Auto-loaded by pytest before any test in tests/ runs — fixtures here are
available by name to every test file with no import needed.
"""

import numpy as np
import pytest


@pytest.fixture
def fake_sbert_model(mocker):
    """
    A fake standing in for sentence_transformers.SentenceTransformer.
    Its .encode() returns a fixed, deterministic array shaped like real
    output (n_sentences, embedding_dim) so downstream code can be tested
    without ever importing the real model.
    """
    fake_model = mocker.MagicMock()

    def fake_encode(sentences, normalize_embeddings=True):
        # Deterministic "embeddings": each sentence maps to one of 4
        # fixed axis-aligned directions, so tests can assert on EXACT
        # similarity values instead of just "some float came back".
        n = len(sentences)
        dim = 4
        vectors = np.eye(dim)[np.arange(n) % dim].astype(float)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vectors = vectors / norms
        return vectors

    fake_model.encode.side_effect = fake_encode
    return fake_model


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Clears embedder.load_model's @lru_cache before every test so a
    mocked (or real) model from one test never leaks into the next."""
    from src.embedder import load_model

    load_model.cache_clear()
    yield
    load_model.cache_clear()
