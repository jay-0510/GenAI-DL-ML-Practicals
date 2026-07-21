"""
utils/config.py
==================
Typed settings for the whole pipeline, loaded from environment
variables / .env. Every module in src/ imports `settings` from here
rather than reading os.environ itself -- one source of truth for region,
bucket name, model IDs, and timing/retry knobs used across S3, Textract,
Bedrock, and OpenSearch calls.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    aws_region: str = "us-east-1"

    # S3 (ingestion/s3_uploader.py)
    s3_bucket_name: str = "jay-patel-378311506052"
    s3_raw_pdf_prefix: str = "raw_pdfs/"

    # Textract (ingestion/text_extractor.py) -- async job polling.
    textract_poll_interval_seconds: int = 5
    textract_timeout_seconds: int = 600

    # Titan Embeddings (embeddings/titan_embeddings.py)
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"

    # OpenSearch (vectorstore/opensearch_store.py)
    opensearch_collection_name: str = "practical8-doc-ingestion"
    opensearch_index_prefix: str = "documents"

    # Chunking (ingestion/chunking.py) -- overlap held constant across
    # the chunk-size experiment so chunk_size is the only variable that
    # changes between the three conditions being compared.
    default_chunk_overlap: int = 50

    max_retry_attempts: int = 3


settings = Settings()
