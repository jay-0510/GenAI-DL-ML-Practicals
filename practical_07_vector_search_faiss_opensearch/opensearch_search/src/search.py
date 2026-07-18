"""
opensearch_search/src/search.py
==================================

WHAT THIS FILE DOES
--------------------
Implements three ways of searching the same 50-document index, so they
can be compared directly:

  1. keyword_search   -- classic lexical (BM25) search on the "text" field.
  2. semantic_search   -- vector similarity search on the "embedding" field.
  3. hybrid_search      -- combines both, via Reciprocal Rank Fusion (RRF).

A WORKED EXAMPLE OF WHY THESE THREE CAN DISAGREE
------------------------------------------------------
The dataset (src/data.py) deliberately includes:
    "This rain-resistant coat kept me completely dry during the storm."
    "This jacket's zipper jammed within the first week of wear."

Searching "waterproof jacket":
  - KEYWORD search matches on literal word overlap. Neither document
    contains "waterproof". The zipper-jamming review DOES contain
    "jacket", so keyword search ranks it highly -- despite that review
    being about a DEFECT, not about waterproofing.
  - SEMANTIC search matches on MEANING. "Rain-resistant" and
    "waterproof" are different words but nearly the same idea, so the
    coat review ranks highly even with zero literal word overlap.
  - HYBRID search (below) tries to get the benefit of both: don't miss
    an obvious literal match, but don't miss an obvious meaning match either.

WHY RECIPROCAL RANK FUSION (RRF), NOT A SINGLE BOOL QUERY WITH BOOSTED
match + knn CLAUSES?
------------------------------------------------------------------------
OpenSearch DOES support putting a `match` clause and a `knn` clause
inside one `bool`/`should` query, and it's tempting to just boost each
clause's weight and let OpenSearch add the scores together. The problem:
BM25 keyword scores are UNBOUNDED (can be small or double digits
depending on term rarity/document length) while cosine/inner-product
scores are bounded to roughly [-1, 1]. Naively summing two scores on
wildly different numeric scales means whichever one happens to have the
larger raw magnitude dominates the ranking -- not because it's more
relevant, but purely because of scale. This is a well-documented pitfall
of naive hybrid search.

RRF sidesteps the problem entirely by combining RANKS (each result's
POSITION in its own list) instead of raw scores -- rank is already on
the same, comparable scale (1st, 2nd, 3rd...) no matter which scoring
system produced it. This is the same core idea OpenSearch's own built-in
"Hybrid Query" feature uses (after score normalization via a search
pipeline); RRF gets a similar effect with a simple client-side
computation and no extra AWS-side pipeline resource to configure --
appropriate for a foundational comparison exercise like this one.
"""

from typing import Dict, List, TypedDict

from opensearchpy import OpenSearch

from utils.config import config


class SearchResult(TypedDict):
    text: str
    category: str
    score: float


def keyword_search(client: OpenSearch, query_text: str, k: int = None) -> List[SearchResult]:
    """
    Classic lexical search: OpenSearch's `match` query scores documents
    by BM25 relevance based on literal word overlap with `query_text`.
    """
    k = k or config.default_top_k
    body = {"size": k, "query": {"match": {"text": query_text}}}
    response = client.search(index=config.index_name, body=body)
    return [
        {"text": hit["_source"]["text"], "category": hit["_source"]["category"], "score": hit["_score"]}
        for hit in response["hits"]["hits"]
    ]


def semantic_search(client: OpenSearch, query_embedding, k: int = None) -> List[SearchResult]:
    """
    Vector similarity search: OpenSearch's `knn` query finds the k
    documents whose stored `embedding` is closest to `query_embedding`
    (by inner product, per the index mapping in indexer.py).
    """
    k = k or config.default_top_k
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": query_embedding.tolist(), "k": k}}},
    }
    response = client.search(index=config.index_name, body=body)
    return [
        {"text": hit["_source"]["text"], "category": hit["_source"]["category"], "score": hit["_score"]}
        for hit in response["hits"]["hits"]
    ]


def _reciprocal_rank_fusion(
    result_lists: List[List[SearchResult]], k_constant: int = 60
) -> List[SearchResult]:
    """
    Merges multiple ranked result lists into one, using Reciprocal Rank
    Fusion: each document's fused score is the sum, across every list it
    appears in, of 1 / (k_constant + rank) -- where `rank` is its
    1-indexed position in that particular list.

    k_constant=60 is the value used in the original RRF paper (Cormack et
    al., 2009) and is a widely-used default -- it controls how much
    influence lower-ranked results still have (a larger constant flattens
    the difference between rank 1 and rank 10; this project just uses the
    standard default rather than tuning it, since re-tuning it isn't the
    point of this exercise).

    Why fuse by TEXT rather than by document ID?
        keyword_search and semantic_search each return their own
        `_source` documents independently -- text is a simple, reliable
        join key across both without needing to also thread `_id` through
        every function signature.
    """
    fused_scores: Dict[str, float] = {}
    doc_lookup: Dict[str, SearchResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            key = result["text"]
            fused_scores[key] = fused_scores.get(key, 0.0) + 1.0 / (k_constant + rank)
            doc_lookup[key] = result

    ranked_keys = sorted(fused_scores, key=lambda key: fused_scores[key], reverse=True)
    return [
        {"text": key, "category": doc_lookup[key]["category"], "score": fused_scores[key]}
        for key in ranked_keys
    ]


def hybrid_search(
    client: OpenSearch, query_text: str, query_embedding, k: int = None
) -> List[SearchResult]:
    """
    Runs keyword_search and semantic_search over a WIDER candidate pool
    than the final result count (config.hybrid_candidate_pool), then
    fuses them with RRF and returns the top `k`.

    Why fetch a wider pool before fusing, rather than just fusing the
    final top-k lists from each?
        A document that's (say) the semantic search's #15 result but
        keyword search's #2 result should still surface in the fused
        top-5 -- but it can only do that if semantic_search actually
        RETURNED 15+ results in the first place. Fusing only two
        already-truncated top-5 lists would silently exclude exactly the
        kind of "good on one signal, not the other" result that hybrid
        search exists to catch.
    """
    k = k or config.default_top_k
    pool_size = config.hybrid_candidate_pool

    keyword_results = keyword_search(client, query_text, k=pool_size)
    semantic_results = semantic_search(client, query_embedding, k=pool_size)

    fused = _reciprocal_rank_fusion([keyword_results, semantic_results])
    return fused[:k]
