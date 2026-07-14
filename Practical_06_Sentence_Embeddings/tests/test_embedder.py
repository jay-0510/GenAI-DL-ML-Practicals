"""
tests/test_embedder.py
=========================
Unit tests for src/embedder.py.

What we verify:
    1. load_model() is cached — calling it twice returns the SAME
       object, meaning the real SentenceTransformer constructor only
       runs once.
    2. encode_sentences() calls the model with normalize_embeddings=True
       (the property similarity.py's dot-product shortcut depends on).
    3. encode_sentences() returns an array shaped (n_sentences, dim).
"""

import numpy as np

from src.embedder import encode_sentences, load_model


def test_load_model_is_cached(mocker):
    mock_constructor = mocker.patch("src.embedder.SentenceTransformer")
    first = load_model()
    second = load_model()
    assert first is second
    mock_constructor.assert_called_once()


def test_encode_sentences_requests_normalized_embeddings(mocker, fake_sbert_model):
    mocker.patch("src.embedder.load_model", return_value=fake_sbert_model)
    encode_sentences(["a", "b"])
    call_kwargs = fake_sbert_model.encode.call_args.kwargs
    assert call_kwargs.get("normalize_embeddings") is True


def test_encode_sentences_returns_correct_shape(mocker, fake_sbert_model):
    mocker.patch("src.embedder.load_model", return_value=fake_sbert_model)
    result = encode_sentences(["one", "two", "three"])
    assert result.shape == (3, 4)  # 4 = fake embedding dim from the fixture
    assert isinstance(result, np.ndarray)
