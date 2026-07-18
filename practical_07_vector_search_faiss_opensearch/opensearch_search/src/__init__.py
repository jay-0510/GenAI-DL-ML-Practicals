"""
src package (opensearch_search)
==================================
AWS half of the practical -- creates a real OpenSearch Serverless vector
search collection and compares three ways of querying it.

    data.py       -> 50 product-review documents across 5 categories
    embedder.py    -> loads SBERT, encodes text into vectors (same model as faiss_search/)
    client.py       -> creates the collection + policies, returns an authenticated client
    indexer.py      -> creates the knn_vector index mapping, bulk-indexes documents
    search.py        -> keyword search, semantic (knn) search, and hybrid search (RRF)
"""
