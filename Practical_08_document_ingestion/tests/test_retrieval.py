"""
tests/test_retrieval.py
==========================
Unit tests for src/retrieval/retriever.py. The underlying vector store
is mocked -- these tests check that this module wires calls through
correctly, not OpenSearch's actual search quality.
"""

from langchain_core.documents import Document

from src.retrieval.retriever import get_retriever, retrieve_with_scores


def test_get_retriever_passes_k_into_search_kwargs(mocker):
    mock_vectorstore = mocker.MagicMock()

    get_retriever(mock_vectorstore, k=7)

    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 7})


def test_retrieve_with_scores_calls_similarity_search_with_query_and_k(mocker):
    fake_results = [(Document(page_content="chunk text"), 0.87)]
    mock_similarity_search = mocker.patch(
        "src.retrieval.retriever.similarity_search", return_value=fake_results
    )
    mock_vectorstore = mocker.MagicMock()

    result = retrieve_with_scores(mock_vectorstore, "what is attention?", k=3)

    mock_similarity_search.assert_called_once_with(mock_vectorstore, "what is attention?", k=3)
    assert result == fake_results
