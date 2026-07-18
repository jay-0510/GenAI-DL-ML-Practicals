"""
opensearch_search/src/client.py
==================================

WHAT THIS FILE DOES
--------------------
Creates an Amazon OpenSearch Serverless VECTOR SEARCH collection (via
boto3's control-plane API), waits for it to become active, and builds an
authenticated opensearch-py client pointed at it for actually
indexing/querying documents.

WHY THREE SEPARATE POLICIES BEFORE THE COLLECTION EVEN EXISTS?
--------------------------------------------------------------------
OpenSearch Serverless requires all of these, and creation fails without
them -- this isn't extra caution on this project's part, it's how AWS
requires the service to be set up:

  1. ENCRYPTION policy -- every collection must have an at-rest encryption
     policy attached before it can be created at all.
  2. NETWORK policy -- controls whether the collection's endpoint (and
     optional dashboards) are reachable from the public internet or only
     from a VPC. We use public access here to keep the practical runnable
     from a laptop with no extra VPC setup.
  3. DATA ACCESS policy -- a separate, IAM-principal-based permission
     grant for who can read/write actual documents. This is deliberately
     independent from standard IAM policies -- AWS designed OpenSearch
     Serverless so collection-level and document-level access are
     controlled here, not just through IAM alone.

WHY boto3 FOR CREATION BUT opensearch-py FOR INDEXING/SEARCH?
--------------------------------------------------------------------
These are two different APIs for two different layers:
  - boto3's `opensearchserverless` client talks to the AWS CONTROL PLANE
    (create/delete collections and policies) -- the same kind of account-
    level management call as, say, creating an S3 bucket.
  - Once the collection exists, actually indexing or querying DOCUMENTS
    inside it goes over OpenSearch's own REST API (index/_doc, index/_search)
    -- boto3 has no client for that. `opensearch-py` is the standard
    Python client for it, and it needs AWS SigV4-signed requests
    (via AWSV4SignerAuth) to authenticate against a Serverless collection.

IMPORTANT: THE SERVICE NAME FOR SIGNING IS "aoss", NOT "es"
--------------------------------------------------------------------
"es" is for classic, managed OpenSearch Service DOMAINS. OpenSearch
Serverless COLLECTIONS use "aoss" instead -- using the wrong one is a
common cause of a 403 Forbidden error that has nothing to do with your
actual permissions.
"""

import time
from typing import Optional

import boto3
import botocore
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection

from utils.config import config

ENCRYPTION_POLICY_NAME = f"{config.collection_name}-enc"
NETWORK_POLICY_NAME = f"{config.collection_name}-net"
DATA_ACCESS_POLICY_NAME = f"{config.collection_name}-access"


def _control_plane_client():
    """boto3 client for collection/policy management (account-level, not
    document-level) -- kept as its own tiny function so every function
    below doesn't repeat `boto3.client('opensearchserverless', ...)`."""
    return boto3.client("opensearchserverless", region_name=config.region_name)


def create_encryption_policy(client) -> None:
    """AWS-owned key encryption for any collection matching this name --
    required before a collection can be created. Swallows ConflictException
    so this function is safe to re-run if the policy already exists from a
    previous run of this notebook."""
    try:
        client.create_security_policy(
            name=ENCRYPTION_POLICY_NAME,
            type="encryption",
            policy=f'{{"Rules":[{{"ResourceType":"collection","Resource":["collection/{config.collection_name}"]}}],"AWSOwnedKey":true}}',
        )
    except client.exceptions.ConflictException:
        pass


def create_network_policy(client) -> None:
    """Allows public access to the collection endpoint. For a real
    production workload you'd typically restrict this to a VPC instead --
    public access is used here so this practical runs from a laptop
    without needing extra VPC/networking setup."""
    try:
        client.create_security_policy(
            name=NETWORK_POLICY_NAME,
            type="network",
            policy=(
                f'[{{"Rules":[{{"ResourceType":"collection","Resource":'
                f'["collection/{config.collection_name}"]}}],"AllowFromPublic":true}}]'
            ),
        )
    except client.exceptions.ConflictException:
        pass


def create_data_access_policy(client, principal_arn: str) -> None:
    """
    Grants a specific IAM principal (user or role) permission to actually
    read/write documents and indexes in this collection.

    Args:
        principal_arn: the ARN of the IAM user/role running this
            notebook. Must be passed explicitly (rather than assumed)
            because this policy is what grants a *different* level of
            access than the caller's normal IAM permissions -- getting
            this wrong is the #1 cause of 403 errors when talking to the
            collection later, so it's made an explicit, visible argument
            instead of something guessed at automatically.
    """
    try:
        client.create_access_policy(
            name=DATA_ACCESS_POLICY_NAME,
            type="data",
            policy=(
                f'[{{"Rules":[{{"ResourceType":"index","Resource":'
                f'["index/{config.collection_name}/*"],"Permission":'
                f'["aoss:CreateIndex","aoss:DeleteIndex","aoss:UpdateIndex",'
                f'"aoss:DescribeIndex","aoss:ReadDocument","aoss:WriteDocument"]}},'
                f'{{"ResourceType":"collection","Resource":["collection/{config.collection_name}"],'
                f'"Permission":["aoss:CreateCollectionItems","aoss:DescribeCollectionItems"]}}],'
                f'"Principal":["{principal_arn}"]}}]'
            ),
        )
    except client.exceptions.ConflictException:
        pass


def create_collection(principal_arn: str) -> str:
    """
    Creates all three required policies, then the VECTOR SEARCH
    collection itself, waits for it to become ACTIVE, and returns its
    endpoint.

    Args:
        principal_arn: your IAM user/role ARN -- find it with
            `aws sts get-caller-identity --query Arn --output text`.

    Why type="VECTORSEARCH" specifically?
        OpenSearch Serverless collections come in three types (SEARCH,
        TIMESERIES, VECTORSEARCH) with different internal optimizations.
        VECTORSEARCH is required to use knn_vector fields at all --
        creating a plain SEARCH collection and trying to add a
        knn_vector field to it will fail.
    """
    client = _control_plane_client()

    create_encryption_policy(client)
    create_network_policy(client)
    create_data_access_policy(client, principal_arn)

    try:
        client.create_collection(name=config.collection_name, type="VECTORSEARCH")
    except client.exceptions.ConflictException:
        pass  # collection already exists from a previous run -- fine, continue

    return _wait_for_active_endpoint(client)


def _wait_for_active_endpoint(client, timeout_seconds: int = 300) -> str:
    """
    Polls the collection's status until it's ACTIVE and returns its
    endpoint.

    Why polling instead of a single describe call? Collection creation is
    asynchronous on AWS's side -- it typically takes a few minutes to
    provision. batch_get_collection returns CREATING until it's ready;
    there's no "wait" API for this, so polling with a short sleep is the
    standard approach (matches the pattern in AWS's own SDK examples).
    """
    start = time.time()
    while time.time() - start < timeout_seconds:
        response = client.batch_get_collection(names=[config.collection_name])
        details = response.get("collectionDetails", [])
        if details and details[0]["status"] == "ACTIVE":
            return details[0]["collectionEndpoint"]
        print("Waiting for collection to become ACTIVE...")
        time.sleep(20)
    raise TimeoutError(
        f"Collection '{config.collection_name}' did not become ACTIVE within {timeout_seconds}s."
    )


def get_opensearch_client(endpoint: str) -> OpenSearch:
    """
    Builds an authenticated opensearch-py client for the data plane
    (indexing/searching documents), pointed at the given collection
    endpoint.

    service="aoss" (not "es") is what makes SigV4 signing work correctly
    against a Serverless collection -- see this module's docstring.
    """
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, config.region_name, "aoss")
    host = endpoint.replace("https://", "")

    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )


def delete_collection() -> None:
    """
    Tears down the collection (and, best-effort, its policies) — call
    this when you're done experimenting.

    WHY THIS FUNCTION EXISTS AND MATTERS: OpenSearch Serverless bills for
    provisioned OpenSearch Compute Units (OCUs) by the hour REGARDLESS OF
    USAGE -- an idle collection left running still costs money. There is
    no free "pause" option; deleting the collection is the only way to
    stop the charge. See the README's cost/cleanup section.
    """
    client = _control_plane_client()
    try:
        client.delete_collection(name=config.collection_name)
        print(f"Deletion requested for collection '{config.collection_name}'.")
    except client.exceptions.ResourceNotFoundException:
        print("Collection already deleted or never created.")

    # Policies are free to leave behind, but cleaning them up avoids
    # naming conflicts if you re-run this notebook later with the same
    # collection_name.
    for policy_name, policy_type in [
        (ENCRYPTION_POLICY_NAME, "encryption"),
        (NETWORK_POLICY_NAME, "network"),
    ]:
        try:
            client.delete_security_policy(name=policy_name, type=policy_type)
        except botocore.exceptions.ClientError:
            pass
    try:
        client.delete_access_policy(name=DATA_ACCESS_POLICY_NAME, type="data")
    except botocore.exceptions.ClientError:
        pass
