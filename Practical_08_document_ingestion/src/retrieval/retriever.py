"""
src/retrieval/retriever.py
=============================
Thin retrieval interface on top of vectorstore.opensearch_store.

WHY IS THIS A SEPARATE MODULE FROM opensearch_store.py?
------------------------------------------------------------
opensearch_store.py's job is "talk to OpenSearch" (indexing, raw
similarity search). This module's job is "answer a retrieval question in
the shape the REST of an application expects" -- e.g. LangChain's
standard `Retriever` interface, which is what you'd plug into a
downstream RAG chain (a later milestone, not this one) without that
chain needing to know anything about OpenSearch specifically. Keeping
the split means swapping the vector store later (OpenSearch -> something
else) would only touch opensearch_store.py, not this module or anything
built on top of it.

NOTE: the actual chunk-size COMPARISON experiment (200 vs 500 vs 1000,
running it across three indexes and writing up findings) lives in the
notebook, not here -- that's a one-off exploratory analysis, not
reusable application logic, which is exactly the kind of thing that
belongs in a notebook rather than src/.
"""

from typing import List, Tuple

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document

from src.vectorstore.opensearch_store import similarity_search


def get_retriever(vectorstore: OpenSearchVectorSearch, k: int = 4):
    """Returns a LangChain Retriever backed by `vectorstore` -- the
    standard interface for plugging this into a RetrievalQA chain or
    similar, without the caller needing to know it's OpenSearch underneath."""
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_with_scores(vectorstore: OpenSearchVectorSearch, query: str, k: int = 4) -> List[Tuple[Document, float]]:
    """Returns the top-k chunks for `query` alongside their similarity
    scores -- exposed here (not just as_retriever(), which drops scores)
    because the chunk-size experiment specifically needs the scores to
    compare retrieval confidence across chunk sizes, not just which
    chunks were returned."""
    return similarity_search(vectorstore, query, k=k)
