"""
tests/test_extractor.py
==========================
Unit tests for src/ingestion/text_extractor.py. All Textract calls are
mocked -- these tests check OUR orchestration logic (pagination,
polling, caching), not Textract's actual OCR quality.
"""

from src.ingestion.text_extractor import (
    blocks_to_text,
    cache_or_extract,
    fetch_all_blocks,
    wait_for_job,
)


def test_blocks_to_text_joins_only_line_blocks_in_order():
    blocks = [
        {"BlockType": "PAGE"},
        {"BlockType": "LINE", "Text": "Attention Is All You Need"},
        {"BlockType": "WORD", "Text": "Attention"},  # should be ignored
        {"BlockType": "LINE", "Text": "Ashish Vaswani, Noam Shazeer"},
    ]
    result = blocks_to_text(blocks)
    assert result == "Attention Is All You Need\nAshish Vaswani, Noam Shazeer"


def test_fetch_all_blocks_follows_next_token_pagination(mocker):
    mock_client = mocker.MagicMock()
    mock_client.get_document_text_detection.side_effect = [
        {"Blocks": [{"BlockType": "LINE", "Text": "page one"}], "NextToken": "abc"},
        {"Blocks": [{"BlockType": "LINE", "Text": "page two"}]},  # no NextToken -> last page
    ]

    blocks = fetch_all_blocks(mock_client, job_id="job-123")

    assert len(blocks) == 2
    assert mock_client.get_document_text_detection.call_count == 2
    # Second call must have passed the NextToken from the first response.
    second_call_kwargs = mock_client.get_document_text_detection.call_args_list[1].kwargs
    assert second_call_kwargs["NextToken"] == "abc"


def test_wait_for_job_returns_on_succeeded(mocker):
    mock_client = mocker.MagicMock()
    mock_client.get_document_text_detection.return_value = {"JobStatus": "SUCCEEDED"}
    wait_for_job(mock_client, job_id="job-123")  # should not raise


def test_wait_for_job_raises_on_failed(mocker):
    mock_client = mocker.MagicMock()
    mock_client.get_document_text_detection.return_value = {
        "JobStatus": "FAILED",
        "StatusMessage": "Unsupported document",
    }
    try:
        wait_for_job(mock_client, job_id="job-123")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Unsupported document" in str(exc)


def test_cache_or_extract_uses_cached_file_without_calling_textract(mocker, tmp_path):
    cache_file = tmp_path / "doc_1_attention.txt"
    cache_file.write_text("cached extracted text", encoding="utf-8")

    extract_spy = mocker.patch("src.ingestion.text_extractor.extract_text_from_s3_pdf")

    result = cache_or_extract("my-bucket", "raw_pdfs/doc_1_attention.pdf", str(cache_file))

    assert result == "cached extracted text"
    extract_spy.assert_not_called()


def test_cache_or_extract_runs_textract_and_writes_cache_when_missing(mocker, tmp_path):
    cache_file = tmp_path / "doc_1_attention.txt"  # does not exist yet
    mocker.patch(
        "src.ingestion.text_extractor.extract_text_from_s3_pdf",
        return_value="freshly extracted text",
    )

    result = cache_or_extract("my-bucket", "raw_pdfs/doc_1_attention.pdf", str(cache_file))

    assert result == "freshly extracted text"
    assert cache_file.read_text(encoding="utf-8") == "freshly extracted text"
