"""
faiss_search/src/faiss_index.py
==================================

WHAT THIS FILE DOES
--------------------
Builds a FAISS index over the sentence embeddings, and searches it for
the nearest neighbors of a query vector.

WHY IndexFlatIP, NOT AN APPROXIMATE INDEX LIKE IVF OR HNSW?
------------------------------------------------------------------
FAISS offers both EXACT indexes (IndexFlat*: compare the query against
every vector, guaranteed correct) and APPROXIMATE indexes (IndexIVFFlat,
IndexHNSWFlat: trade a small amount of accuracy for much faster search at
large scale). Approximate indexes exist to solve a problem this practical
doesn't have -- at 100 vectors, an exhaustive comparison is already
sub-millisecond, so there's no speed problem to trade accuracy for.
Approximate indexes also need extra tuning (e.g. IVF needs a training
step and a choice of cluster count) that would add complexity with zero
practical benefit at this scale. IndexFlatIP is the right, boring choice
here -- these indexes become worth it once you're indexing hundreds of
thousands to millions of vectors.

WHY IndexFlatIP (INNER PRODUCT), NOT IndexFlatL2 (EUCLIDEAN DISTANCE)?
----------------------------------------------------------------------------
embedder.py normalizes every embedding to unit length. For unit vectors,
the inner product between two vectors IS their cosine similarity -- so
IndexFlatIP gives us exact cosine-similarity search "for free," without
needing to implement cosine similarity by hand or post-process L2
distances into a similarity score. This mirrors the same normalize-then-
dot-product trade-off made in Practical 6's similarity.py.
"""

from typing import List, Tuple

import faiss
import numpy as np


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds an exact, inner-product FAISS index over the given embeddings.

    Args:
        embeddings: (n, dim) float32 array of NORMALIZED embeddings (see
            embedder.py -- normalization is what makes inner product
            equivalent to cosine similarity here).

    Returns:
        A FAISS index with all `n` vectors already added, ready to search.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search(
    index: faiss.Index, query_embedding: np.ndarray, sentences: List[str], k: int = 5
) -> List[Tuple[str, float]]:
    """
    Finds the k nearest neighbors of a single query embedding.

    Args:
        index: a FAISS index built by build_faiss_index().
        query_embedding: a single (dim,) normalized embedding vector.
        sentences: the original sentence text, in the SAME order the
            embeddings were added to the index in -- FAISS returns
            positional indices, not the text itself, so this list is what
            translates "neighbor #37" back into an actual sentence.
        k: how many neighbors to return.

    Returns:
        A list of (sentence, similarity_score) tuples, best match first.
    """
    # FAISS's search() expects a 2D batch of queries, shape (n_queries, dim)
    # -- we're only searching one query at a time here, so wrap it in an
    # extra dimension and unwrap the single row of results afterward.
    query_batch = np.asarray([query_embedding], dtype=np.float32)
    scores, indices = index.search(query_batch, k)

    return [(sentences[idx], float(score)) for idx, score in zip(indices[0], scores[0])]


if __name__ == "__main__":
    # Manual smoke test: `python -m src.faiss_index`
    from src.data import get_sentences
    from src.embedder import encode_sentences

    sentences = get_sentences()
    embeddings = encode_sentences(sentences)
    index = build_faiss_index(embeddings)

    query = "The chef added fresh herbs to the pasta."
    query_embedding = encode_sentences([query])[0]
    results = search(index, query_embedding, sentences, k=5)

    print(f"Query: {query}\n")
    for rank, (sentence, score) in enumerate(results, start=1):
        print(f"{rank}. ({score:.3f}) {sentence}")
