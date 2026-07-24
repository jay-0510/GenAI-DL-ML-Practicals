"""
src/vectorstore/opensearch_store.py
======================================
Two layers, in one file since this practical has only one vectorstore
module (unlike Practical 7, which split collection provisioning into its
own client.py):

  1. CONTROL PLANE (boto3 `opensearchserverless`) -- creates the
     encryption/network/data-access policies and the VECTORSEARCH
     collection itself, and waits for it to become ACTIVE.
  2. DATA PLANE (langchain_community's OpenSearchVectorSearch) -- once
     the collection is active, this wraps it as a LangChain vector store
     so chunks can be indexed and queried through LangChain's standard
     interface instead of hand-writing raw OpenSearch request bodies.

WHY requests_aws4auth.AWS4Auth, NOT opensearch-py's OWN AWSV4SignerAuth?
------------------------------------------------------------------------------
LangChain's OpenSearchVectorSearch auto-detects whether it's talking to
an AOSS (Serverless) collection by inspecting the `http_auth` object it's
given, and there's a documented LangChain bug (GitHub issue #14129) where
opensearch-py's own AWSV4SignerAuth isn't reliably detected as AOSS by
that check, silently causing AOSS-incompatible requests (like `refresh`)
to be sent. `requests_aws4auth.AWS4Auth` with `service="aoss"` is the
version demonstrated in AWS's own official Bedrock sample notebooks and
is reliably detected -- so it's used here specifically to sidestep a
known, documented compatibility gap, not out of general preference.
"""

import time
from typing import List, Optional

import boto3
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_ENCRYPTION_POLICY_NAME = f"{settings.opensearch_collection_name}-enc"
_NETWORK_POLICY_NAME = f"{settings.opensearch_collection_name}-net"
_DATA_ACCESS_POLICY_NAME = f"{settings.opensearch_collection_name}-access"


# ---------------------------------------------------------------------------
# Control plane: collection + required policies
# ---------------------------------------------------------------------------
def create_collection(principal_arn: str) -> str:
    """
    Creates the three required policies (encryption, network, data
    access -- all mandatory before a collection can exist at all, per
    AWS's own requirements) and the VECTORSEARCH collection, then waits
    for it to become ACTIVE.

    Args:
        principal_arn: your IAM user/role ARN, granted access via the
            data access policy. Find it with
            `aws sts get-caller-identity --query Arn --output text`.

    Returns:
        The collection's HTTPS endpoint.
    """
    client = boto3.client("opensearchserverless", region_name=settings.aws_region)

    try:
        client.create_security_policy(
            name=_ENCRYPTION_POLICY_NAME,
            type="encryption",
            policy=(
                f'{{"Rules":[{{"ResourceType":"collection","Resource":'
                f'["collection/{settings.opensearch_collection_name}"]}}],"AWSOwnedKey":true}}'
            ),
        )
    except client.exceptions.ConflictException:
        pass

    try:
        client.create_security_policy(
            name=_NETWORK_POLICY_NAME,
            type="network",
            # Public access, for a setup runnable from a laptop with no
            # extra VPC configuration -- restrict this to a VPC for
            # anything beyond learning/experimentation.
            policy=(
                f'[{{"Rules":[{{"ResourceType":"collection","Resource":'
                f'["collection/{settings.opensearch_collection_name}"]}}],"AllowFromPublic":true}}]'
            ),
        )
    except client.exceptions.ConflictException:
        pass

    try:
        client.create_access_policy(
            name=_DATA_ACCESS_POLICY_NAME,
            type="data",
            policy=(
                f'[{{"Rules":[{{"ResourceType":"index","Resource":'
                f'["index/{settings.opensearch_collection_name}/*"],"Permission":'
                f'["aoss:CreateIndex","aoss:DeleteIndex","aoss:UpdateIndex",'
                f'"aoss:DescribeIndex","aoss:ReadDocument","aoss:WriteDocument"]}},'
                f'{{"ResourceType":"collection","Resource":'
                f'["collection/{settings.opensearch_collection_name}"],"Permission":'
                f'["aoss:CreateCollectionItems","aoss:DescribeCollectionItems"]}}],'
                f'"Principal":["{principal_arn}"]}}]'
            ),
        )
    except client.exceptions.ConflictException:
        pass

    try:
        client.create_collection(name=settings.opensearch_collection_name, type="VECTORSEARCH")
    except client.exceptions.ConflictException:
        pass  # already exists from a previous run

    return _wait_for_active_endpoint(client)


def _wait_for_active_endpoint(client, timeout_seconds: int = 300) -> str:
    """Polls until the collection is ACTIVE and returns its endpoint --
    collection creation is asynchronous on AWS's side and typically takes
    a few minutes."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        response = client.batch_get_collection(names=[settings.opensearch_collection_name])
        details = response.get("collectionDetails", [])
        if details and details[0]["status"] == "ACTIVE":
            return details[0]["collectionEndpoint"]
        logger.info("Waiting for collection to become ACTIVE...")
        time.sleep(20)
    raise TimeoutError(f"Collection did not become ACTIVE within {timeout_seconds}s.")


def delete_collection() -> None:
    """
    Tears down the collection -- call this when you're done
    experimenting. OpenSearch Serverless bills hourly for provisioned
    compute REGARDLESS OF USAGE; there's no "pause," only delete. See
    the README's cost warning.
    """
    client = boto3.client("opensearchserverless", region_name=settings.aws_region)
    try:
        client.delete_collection(name=settings.opensearch_collection_name)
        logger.info(f"Deletion requested for collection '{settings.opensearch_collection_name}'.")
    except client.exceptions.ResourceNotFoundException:
        logger.info("Collection already deleted or never created.")


# ---------------------------------------------------------------------------
# Data plane: LangChain vector store wiring
# ---------------------------------------------------------------------------
def get_vectorstore(collection_endpoint: str, index_name: str, embeddings: Embeddings) -> OpenSearchVectorSearch:
    """
    Wraps an (already-ACTIVE) collection as a LangChain OpenSearchVectorSearch.

    engine="faiss" is required, not a preference: OpenSearch Serverless
    VECTORSEARCH collections only support the HNSW algorithm using the
    Faiss engine internally -- other combinations that work on
    self-managed OpenSearch clusters aren't available in Serverless.

    Args:
        index_name: lets the SAME collection hold multiple independent
            indexes -- used by the chunk-size experiment to create
            separate indexes per chunk size (e.g. "documents-chunk200",
            "documents-chunk500") within one collection, rather than
            provisioning a whole new (billed) collection per experiment.
    """
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(region=settings.aws_region, service="aoss", refreshable_credentials=credentials)

    return OpenSearchVectorSearch(
        opensearch_url=collection_endpoint,
        index_name=index_name,
        embedding_function=embeddings,
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        engine="faiss",
        timeout=300,
    )


def index_documents(vectorstore: OpenSearchVectorSearch, documents: List[Document]) -> List[str]:
    """
    Embeds and indexes a list of chunked Documents. Thin wrapper over
    `add_documents()` -- kept as its own function mainly so callers don't
    need to know LangChain's exact method name, and so this is the one
    place a retry/error-handling policy would be added later if needed.
    """
    ids = vectorstore.add_documents(documents)
    logger.info(f"Indexed {len(ids)} chunks into '{vectorstore.index_name}'.")
    return ids


def similarity_search(
    vectorstore: OpenSearchVectorSearch, query: str, k: int = 4
) -> List[tuple]:
    """Returns the k most similar chunks to `query`, each paired with its
    similarity score -- the score (not just the ranked list) is what lets
    the chunk-size experiment compare retrieval CONFIDENCE, not just
    which chunks came back."""
    return vectorstore.similarity_search_with_score(query, k=k)
