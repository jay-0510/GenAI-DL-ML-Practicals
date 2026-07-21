"""
tests/test_embeddings.py
===========================
Unit tests for src/embeddings/titan_embeddings.py.
"""

from src.embeddings.titan_embeddings import get_titan_embeddings
from utils.config import settings


def test_get_titan_embeddings_uses_configured_model_id():
    embeddings = get_titan_embeddings()
    assert embeddings.model_id == settings.embedding_model_id


def test_get_titan_embeddings_uses_configured_region():
    embeddings = get_titan_embeddings()
    assert embeddings.region_name == settings.aws_region


def test_get_titan_embeddings_returns_a_new_instance_each_call():
    # Deliberately NOT cached (see the module's docstring for why) --
    # confirms that choice rather than assuming it.
    first = get_titan_embeddings()
    second = get_titan_embeddings()
    assert first is not second
