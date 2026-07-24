"""
src/ingestion/text_extractor.py
==================================
Extracts text from S3-stored PDFs using Textract's ASYNCHRONOUS API, and
caches results to data/extracted_text/ so re-running the notebook
doesn't re-pay for (or re-wait on) Textract on documents already processed.

WHY THE ASYNCHRONOUS API, NOT THE SYNCHRONOUS DetectDocumentText?
------------------------------------------------------------------------
Textract's synchronous DetectDocumentText DOES accept PDFs -- but only
SINGLE-PAGE ones, capped at 10MB (confirmed directly against AWS's own
FAQ and re:Post answers). Our 5 source documents are realistically
multi-page (research papers, docs), so the synchronous API would fail
outright past page 1 of any of them. The asynchronous API
(StartDocumentTextDetection + GetDocumentTextDetection) supports up to
1,000 pages and 500MB, and is the AWS-documented, REQUIRED path for
multi-page PDFs -- this is a hard capability difference, not a style
preference.

WHY CACHE TO data/extracted_text/*.txt?
------------------------------------------
Textract billing is per-page, and the async job can take anywhere from
seconds to a couple of minutes per document. Re-running this notebook
from scratch every time you want to test something downstream (chunking,
embeddings) would re-pay that cost/wait for no reason -- caching the
extracted plain text locally means Textract only runs once per document,
ever, unless you delete the cache file yourself.
"""

import time
from pathlib import Path
from typing import List

import boto3

from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def start_text_detection(textract_client, bucket: str, key: str) -> str:
    """Starts an async Textract job for an S3-stored document and
    returns its JobId. Returns immediately -- the job runs in the
    background on AWS's side."""
    response = textract_client.start_document_text_detection(
        DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}}
    )
    return response["JobId"]


def wait_for_job(textract_client, job_id: str) -> None:
    """
    Polls GetDocumentTextDetection until the job reaches a terminal
    status (SUCCEEDED or FAILED).

    Why polling instead of the SNS-notification approach AWS also
    supports? SNS requires provisioning an SNS topic + IAM role just for
    job-completion notifications -- extra AWS setup unrelated to what
    this practical is teaching. Polling is simpler and sufficient at the
    scale of 5 documents; SNS becomes worth the setup at higher volume /
    when you need to avoid busy-waiting entirely.
    """
    start = time.time()
    while time.time() - start < settings.textract_timeout_seconds:
        response = textract_client.get_document_text_detection(JobId=job_id)
        status = response["JobStatus"]
        if status == "SUCCEEDED":
            return
        if status == "FAILED":
            raise RuntimeError(f"Textract job {job_id} failed: {response.get('StatusMessage')}")
        time.sleep(settings.textract_poll_interval_seconds)
    raise TimeoutError(f"Textract job {job_id} did not complete within {settings.textract_timeout_seconds}s.")


def fetch_all_blocks(textract_client, job_id: str) -> List[dict]:
    """
    Fetches every Block from a completed job, following pagination.

    Why does this need pagination handling at all? GetDocumentTextDetection
    caps how many Blocks it returns per call -- a multi-page document can
    easily produce more LINE blocks than fit in one response. NextToken
    is how Textract tells us there's more to fetch; without following it,
    longer documents would silently come back truncated.
    """
    blocks: List[dict] = []
    next_token = None
    while True:
        kwargs = {"JobId": job_id}
        if next_token:
            kwargs["NextToken"] = next_token
        response = textract_client.get_document_text_detection(**kwargs)
        blocks.extend(response.get("Blocks", []))
        next_token = response.get("NextToken")
        if not next_token:
            break
    return blocks


def blocks_to_text(blocks: List[dict]) -> str:
    """
    Joins Textract's LINE blocks into plain text, one line per line.

    Only LINE blocks are used (not WORD or PAGE) -- LINE blocks already
    represent Textract's own reading-order reconstruction of each line,
    so re-deriving that from individual WORD blocks would be duplicating
    work Textract already did correctly.
    """
    lines = [block["Text"] for block in blocks if block.get("BlockType") == "LINE"]
    return "\n".join(lines)


def extract_text_from_s3_pdf(bucket: str, key: str) -> str:
    """Runs the full async extraction flow for one S3-stored PDF and
    returns its plain text. This is the one function callers outside
    this module should normally use directly."""
    textract_client = boto3.client("textract", region_name=settings.aws_region)
    job_id = start_text_detection(textract_client, bucket, key)
    logger.info(f"Started Textract job {job_id} for s3://{bucket}/{key}")
    wait_for_job(textract_client, job_id)
    blocks = fetch_all_blocks(textract_client, job_id)
    return blocks_to_text(blocks)


def cache_or_extract(bucket: str, key: str, cache_path: str) -> str:
    """
    Returns the cached extracted text at `cache_path` if it already
    exists; otherwise runs Textract and writes the result to that path
    for next time.

    This is the function the notebook actually calls -- extract_text_from_s3_pdf()
    stays a "pure" extraction function with no filesystem side effects,
    while this wrapper owns the caching policy on top of it.
    """
    path = Path(cache_path)
    if path.exists():
        logger.info(f"Using cached extracted text: {cache_path}")
        return path.read_text(encoding="utf-8")

    text = extract_text_from_s3_pdf(bucket, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info(f"Cached extracted text -> {cache_path}")
    return text
