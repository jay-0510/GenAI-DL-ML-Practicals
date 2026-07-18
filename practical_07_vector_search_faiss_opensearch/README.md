# Practical 7 — Vector Search: FAISS to OpenSearch

Two independent, modular sub-projects sharing one notebook:

1. **`faiss_search/`** — index 100 sentence embeddings locally with FAISS
   and run similarity search. No AWS account needed.
2. **`opensearch_search/`** — create a real Amazon OpenSearch Serverless
   vector search collection with boto3, index 50 documents (text +
   embeddings), and compare keyword, semantic, and hybrid search.

---

## Project structure

```
practical_07_faiss_opensearch/
├── README.md
├── requirements.txt
├── .gitignore
├── notebook/
│   └── faiss_to_opensearch.ipynb     # narrated walkthrough of both sub-projects
├── faiss_search/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data.py                    # 100 labeled example sentences
│   │   ├── embedder.py                # loads SBERT, encodes sentences
│   │   └── faiss_index.py             # builds a FAISS index, runs similarity search
│   └── utils/
│       ├── __init__.py
│       └── config.py                  # embedding model, default top_k
└── opensearch_search/
    ├── src/
    │   ├── __init__.py
    │   ├── data.py                    # 50 product-review documents, 5 categories
    │   ├── embedder.py                # same SBERT model, for a fair comparison
    │   ├── client.py                  # creates the collection + policies, returns a client
    │   ├── indexer.py                 # creates the knn_vector index, bulk-indexes docs
    │   └── search.py                  # keyword, semantic, and hybrid (RRF) search
    └── utils/
        ├── __init__.py
        └── config.py                  # AWS region, collection/index names, embedding dim
```
---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Part 1 (FAISS) needs nothing else
Runs entirely locally after the one-time SBERT model download.

### 3. Part 2 (OpenSearch) needs AWS credentials + your principal ARN
```bash
aws configure
```
Find your IAM principal ARN (needed when creating the collection's data
access policy):
```bash
aws sts get-caller-identity --query Arn --output text
```
Required permissions: `opensearchserverless:*` (create/manage
collections and policies) at minimum for the setup steps in this
notebook — scope this down for anything beyond learning/experimentation.

### 4. (Optional) override defaults via `.env` in `opensearch_search/`
```
AWS_REGION=us-east-1
OPENSEARCH_COLLECTION_NAME=practical7-vector-search
OPENSEARCH_INDEX_NAME=documents
```

---

## Usage

### Run the notebook (recommended)
```bash
cd notebook
jupyter notebook faiss_to_opensearch.ipynb
```

### Or run each sub-project directly
```bash
cd faiss_search && python -m src.faiss_index      # local, no AWS
cd ../opensearch_search && python -m src.embedder  # smoke test, no AWS calls
```
`client.py`, `indexer.py`, and `search.py` are meant to be driven from
the notebook (or your own script) since they need a live collection
endpoint and IAM principal ARN as arguments — they're not designed to be
run standalone with `python -m`.

---

## Task 1 summary: local FAISS search

**Flow:** encode 100 labeled sentences with SBERT → build an exact FAISS
index → for a new query, encode it the same way and find its nearest
neighbors.


## Task 2 summary: AWS OpenSearch Serverless

**Flow:** create an encryption policy, network policy, and data access
policy (all required before a collection can exist) → create a
`VECTORSEARCH` collection → wait for it to become `ACTIVE` → create an
index with a `knn_vector` field → bulk-index 50 documents (text +
category + embedding) → run keyword, semantic, and hybrid queries.

---
