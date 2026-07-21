"""
src/ingestion/chunking.py
============================
Wraps LangChain's RecursiveCharacterTextSplitter to turn extracted
document text into retrieval-sized chunks.

WHY RecursiveCharacterTextSplitter SPECIFICALLY?
------------------------------------------------------
It splits on a PRIORITIZED list of separators (paragraph breaks, then
line breaks, then sentences, then words) and only falls back to a
cruder split when a chunk still doesn't fit -- so chunks tend to break at
natural text boundaries instead of mid-sentence or mid-word. This is
LangChain's own documented recommendation as the default choice for
generic text, which is exactly what these 5 documents are (papers,
prose-heavy docs) -- no structural format (code, markdown headers, etc.)
that would call for one of the more specialized splitters instead.

WHY IS chunk_overlap A SEPARATE ARGUMENT FROM chunk_size, NOT A FIXED
PROPORTION OF IT?
------------------------------------------------------------------------
Overlap exists to avoid losing context AT a chunk boundary (a sentence
split across two chunks loses meaning in both halves otherwise). Keeping
it as an independent, FIXED value (see utils/config.py's
default_chunk_overlap) while chunk_size varies across the 200/500/1000
experiment means chunk_size is the ONLY variable changing between the
three conditions being compared -- if overlap scaled with chunk_size too,
any difference in retrieval quality couldn't be cleanly attributed to
chunk size alone.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import settings


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> List[Document]:
    """
    Splits `text` into a list of LangChain Document chunks.

    Args:
        text: the full extracted document text.
        chunk_size: max characters per chunk (this is the variable under
            test in the 200/500/1000 experiment).
        chunk_overlap: characters shared between consecutive chunks.
            Defaults to settings.default_chunk_overlap when not given, so
            callers only need to pass it explicitly if deliberately
            testing overlap itself rather than chunk_size.
        metadata: attached to every resulting chunk (e.g. {"source":
            "doc_1_attention.pdf", "chunk_size": 200}) -- this is what
            lets retrieval results be traced back to which document AND
            which chunk-size experiment they came from.

    Returns:
        A list of Document objects (LangChain's standard unit for
        anything that gets embedded/stored), ready to pass to
        vectorstore.opensearch_store's indexing functions.
    """
    overlap = chunk_overlap if chunk_overlap is not None else settings.default_chunk_overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.create_documents([text], metadatas=[metadata or {}])
