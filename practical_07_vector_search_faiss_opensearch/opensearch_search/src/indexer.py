"""
opensearch_search/src/indexer.py
===================================

WHAT THIS FILE DOES
--------------------
Creates an OpenSearch index with a knn_vector field mapping, and
bulk-indexes the 50 documents (text + category + embedding) into it.

WHY "hnsw" + "faiss" SPECIFICALLY FOR THE knn_vector METHOD?
------------------------------------------------------------------
OpenSearch Serverless VECTOR SEARCH collections only support the HNSW
algorithm using the Faiss engine -- other combinations (e.g. IVF, or the
nmslib engine) that work on self-managed OpenSearch clusters aren't
available in Serverless vector collections. This isn't a stylistic
choice here; it's the only supported option.

WHY space_type="innerproduct", NOT THE MORE COMMONLY-SHOWN "l2" OR
"cosinesimil"?
------------------------------------------------------------------------
embedder.py already normalizes every embedding to unit length. For unit
vectors, inner product IS cosine similarity -- and OpenSearch's own docs
note that "innerproduct" computes faster than "cosinesimil" precisely
because it skips the extra normalization step "cosinesimil" would
otherwise do internally. Since normalization already happened at encode
time, asking OpenSearch to do it again would be redundant work for the
same result. This mirrors the identical trade-off already made in
faiss_search/ (IndexFlatIP over IndexFlatL2) -- same underlying math,
same reasoning, different infrastructure.
"""

from typing import List

from opensearchpy import OpenSearch
from opensearchpy.helpers import bulk

from src.data import Document
from utils.config import config


def create_index_with_knn_mapping(client: OpenSearch) -> None:
    """
    Creates the index with a knn_vector field ("embedding") alongside
    ordinary text/keyword fields for "text" and "category".

    "text" is mapped as `text` (analyzed, tokenized -- what keyword_search
    in search.py runs `match` queries against) and "category" as
    `keyword` (exact-match, unanalyzed -- suitable for filtering/faceting,
    not for full-text search) -- these are OpenSearch's standard field
    types for "search this" vs. "filter/group by this" respectively.

    Safe to call even if the index already exists from a previous run --
    OpenSearch returns a resource_already_exists_exception, which is
    caught and ignored so this notebook can be re-run without manual
    cleanup first.
    """
    index_body = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "category": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": config.embedding_dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "faiss",
                        "space_type": "innerproduct",
                    },
                },
            }
        },
    }

    try:
        client.indices.create(index=config.index_name, body=index_body)
        print(f"Created index '{config.index_name}'.")
    except Exception as exc:
        if "resource_already_exists_exception" in str(exc):
            print(f"Index '{config.index_name}' already exists -- reusing it.")
        else:
            raise


def index_documents(client: OpenSearch, documents: List[Document], embeddings) -> None:
    """
    Bulk-indexes all documents in one round trip.

    Why bulk() instead of calling client.index() once per document?
        A single bulk request means 1 network round-trip instead of 50 --
        for 50 documents this is a meaningful speed difference, and it's
        the standard, recommended way to load more than a handful of
        documents into OpenSearch (indexing one at a time is fine for a
        single document, but doesn't scale).

    Args:
        documents: from src.data.get_documents(), in a fixed order.
        embeddings: (n, dim) array from embedder.encode_texts(), where
            row i corresponds to documents[i] -- position-matched, the
            same pattern used throughout this repo's other practicals.
    """
    actions = [
    {
        "_index": config.index_name,
        "_source": {
            "text": doc["text"],
            "category": doc["category"],
            "embedding": embedding.tolist(),
        },
    }
    for doc, embedding in zip(documents, embeddings)
]

    success_count, errors = bulk(client, actions)
    print(f"Indexed {success_count} documents.")
    if errors:
        print(f"Encountered {len(errors)} errors during indexing:", errors)
