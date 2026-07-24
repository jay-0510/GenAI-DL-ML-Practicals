"""
test_summarize.py
------------------
Covers the /summarize endpoint's happy path and response contract.
"""


def test_summarize_returns_expected_format(client):
    """Successful summarization should return summary text plus length metrics."""
    long_text = "word " * 50
    response = client.post("/summarize", json={"text": long_text, "max_length": 10})

    assert response.status_code == 200
    body = response.json()
    assert "summary" in body
    assert "original_length" in body
    assert "summary_length" in body
    assert body["original_length"] == 50


def test_summarize_uses_default_max_length_when_omitted(client):
    """max_length is optional in the request model and should default to 100."""
    response = client.post("/summarize", json={"text": "Some reasonably long text to summarize."})

    assert response.status_code == 200
