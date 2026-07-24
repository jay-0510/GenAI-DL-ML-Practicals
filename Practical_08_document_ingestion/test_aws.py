"""
test_aws.py
=============
A standalone AWS connectivity smoke-test script -- NOT part of the
automated pytest suite (that lives in tests/, and only runs against
mocked AWS calls). Despite the filename, this contains no `test_*`
functions, so pytest would collect zero tests from it even if it were
ever accidentally included in a pytest run.

WHY THIS EXISTS
-----------------
This pipeline touches FOUR separate AWS services (S3, Textract, Bedrock,
OpenSearch Serverless) before you ever get to the actual notebook -- if
credentials or permissions are misconfigured, it's much faster to find
that out from one script with clear pass/fail output than from a
confusing failure three cells into the notebook.

Run it directly:
    python test_aws.py
"""

import sys

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from utils.config import settings


def check_credentials() -> bool:
    try:
        identity = boto3.client("sts", region_name=settings.aws_region).get_caller_identity()
        print(f"  Credentials OK -- identity: {identity['Arn']}")
        return True
    except (NoCredentialsError, ClientError) as exc:
        print(f"  FAILED: {exc}")
        return False


def check_s3() -> bool:
    try:
        boto3.client("s3", region_name=settings.aws_region).list_buckets()
        print("  S3 reachable.")
        return True
    except ClientError as exc:
        print(f"  FAILED: {exc}")
        return False


def check_textract() -> bool:
    try:
        # No cheap "ping" call exists for Textract; constructing the
        # client and letting boto3 resolve the endpoint/region is enough
        # to catch a bad region or missing service availability.
        boto3.client("textract", region_name=settings.aws_region)
        print("  Textract client constructed OK.")
        return True
    except ClientError as exc:
        print(f"  FAILED: {exc}")
        return False


def check_bedrock_model_access() -> bool:
    try:
        client = boto3.client("bedrock", region_name=settings.aws_region)
        client.get_foundation_model(modelIdentifier=settings.embedding_model_id)
        print(f"  Bedrock model '{settings.embedding_model_id}' is accessible.")
        return True
    except ClientError as exc:
        print(f"  FAILED (check Model Access in the Bedrock console): {exc}")
        return False


def check_opensearch_serverless() -> bool:
    try:
        boto3.client("opensearchserverless", region_name=settings.aws_region).list_collections()
        print("  OpenSearch Serverless reachable.")
        return True
    except ClientError as exc:
        print(f"  FAILED: {exc}")
        return False


def main() -> None:
    checks = [
        ("AWS credentials", check_credentials),
        ("S3", check_s3),
        ("Textract", check_textract),
        ("Bedrock model access", check_bedrock_model_access),
        ("OpenSearch Serverless", check_opensearch_serverless),
    ]

    print(f"Checking AWS connectivity in region: {settings.aws_region}\n")
    results = []
    for name, check_fn in checks:
        print(f"[{name}]")
        results.append(check_fn())
        print()

    if all(results):
        print("All checks passed -- ready to run the notebook.")
    else:
        print("One or more checks failed -- fix these before running the notebook.")
        sys.exit(1)


if __name__ == "__main__":
    main()
