"""
src/ingestion/s3_uploader.py
===============================
Uploads local PDFs to S3.

WHY DOES A LOCAL PDF NEED TO GO TO S3 AT ALL, BEFORE TEXT EXTRACTION?
--------------------------------------------------------------------------
Textract's asynchronous API (required for multi-page PDFs -- see
text_extractor.py's docstring for why) only accepts documents that are
ALREADY in S3, referenced by bucket/key -- unlike the synchronous API,
it has no option to send raw file bytes directly in the request. So S3
upload isn't an optional convenience step here; it's a hard prerequisite
for the extraction method this pipeline needs to use.
"""

from pathlib import Path
from typing import Dict
import boto3
from botocore.exceptions import ClientError

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def ensure_bucket_exists(s3_client) -> None:
    """
    Ensures the configured bucket exists and is accessible.

    Handles global naming collisions (403 Forbidden) and prevents
    failing when the bucket is already owned by the current account.
    """
    try:
        s3_client.head_bucket(Bucket=settings.s3_bucket_name)
        logger.info(f"Bucket '{settings.s3_bucket_name}' already exists and is accessible.")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")

        # 403 Forbidden: Bucket exists globally but belongs to someone else
        if error_code == "403":
            logger.critical(
                f"Bucket name collision! The name '{settings.s3_bucket_name}' "
                f"is already taken by another AWS account globally."
            )
            raise RuntimeError(
                f"Global S3 collision: Change your 's3_bucket_name' in your config/settings."
            ) from e

        # 404 Not Found: Bucket does not exist anywhere, safe to create
        elif error_code == "404":
            create_kwargs = {"Bucket": settings.s3_bucket_name}

            # us-east-1 inconsistency guard
            if settings.aws_region != "us-east-1":
                create_kwargs["CreateBucketConfiguration"] = {
                    "LocationConstraint": settings.aws_region
                }

            try:
                s3_client.create_bucket(**create_kwargs)
                logger.info(f"Successfully created new bucket '{settings.s3_bucket_name}'.")
            except ClientError as ce:
                # Fallback catch for race conditions or unexpected creation errors
                if ce.response.get("Error", {}).get("Code") == "BucketAlreadyExists":
                    raise RuntimeError(
                        f"The bucket name '{settings.s3_bucket_name}' is globally unavailable."
                    ) from ce
                raise ce
        else:
            # Re-raise any other unexpected AWS client errors (e.g., invalid credentials)
            raise e


def upload_pdfs(local_dir: str) -> Dict[str, str]:
    """
    Uploads every .pdf file in `local_dir` to S3 under the configured
    prefix.

    Returns:
        {filename: s3_key} for every uploaded file -- text_extractor.py
        needs the s3_key (not just the filename) to start a Textract job.
    """
    s3_client = boto3.client("s3", region_name=settings.aws_region)
    ensure_bucket_exists(s3_client)

    pdf_paths = sorted(Path(local_dir).glob("*.pdf"))
    if not pdf_paths:
        logger.warning(f"No PDFs found in {local_dir} -- nothing to upload.")

    uploaded = {}
    for pdf_path in pdf_paths:
        s3_key = f"{settings.s3_raw_pdf_prefix}{pdf_path.name}"
        s3_client.upload_file(str(pdf_path), settings.s3_bucket_name, s3_key)
        uploaded[pdf_path.name] = s3_key
        logger.info(f"Uploaded {pdf_path.name} -> s3://{settings.s3_bucket_name}/{s3_key}")

    return uploaded
