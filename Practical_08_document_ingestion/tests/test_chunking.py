"""
tests/test_chunking.py
=========================
Unit tests for src/ingestion/chunking.py -- pure text-splitting logic,
no AWS calls, no mocking needed.
"""

from src.ingestion.chunking import chunk_text

# Long enough to force multiple chunks at small chunk_size values below.
SAMPLE_TEXT = (
    "Attention mechanisms have become an integral part of compelling sequence "
    "modeling and transduction models in various tasks. They allow modeling of "
    "dependencies without regard to their distance in the input or output "
    "sequences. In this work we propose a new simple network architecture, the "
    "Transformer, based solely on attention mechanisms, dispensing with "
    "recurrence and convolutions entirely."
)


def test_chunk_text_respects_chunk_size_upper_bound():
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    for chunk in chunks:
        # RecursiveCharacterTextSplitter tries to break on natural
        # boundaries, so individual chunks can occasionally land under
        # the limit but should never meaningfully exceed it.
        assert len(chunk.page_content) <= 120


def test_smaller_chunk_size_produces_more_chunks():
    small_chunks = chunk_text(SAMPLE_TEXT, chunk_size=50, chunk_overlap=0)
    large_chunks = chunk_text(SAMPLE_TEXT, chunk_size=500, chunk_overlap=0)
    assert len(small_chunks) > len(large_chunks)


def test_chunk_text_attaches_metadata_to_every_chunk():
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=100, metadata={"source": "doc_1_attention.pdf"})
    assert all(chunk.metadata.get("source") == "doc_1_attention.pdf" for chunk in chunks)


def test_chunk_overlap_defaults_from_settings_when_not_given(mocker):
    mocker.patch("src.ingestion.chunking.settings.default_chunk_overlap", 25)
    splitter_spy = mocker.patch("src.ingestion.chunking.RecursiveCharacterTextSplitter")
    splitter_spy.return_value.create_documents.return_value = []

    chunk_text(SAMPLE_TEXT, chunk_size=200)  # chunk_overlap intentionally omitted

    splitter_spy.assert_called_once_with(chunk_size=200, chunk_overlap=25)
