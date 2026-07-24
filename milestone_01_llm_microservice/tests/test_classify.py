"""
test_classify.py
-----------------
Covers the /classify endpoint's happy path and response contract.
"""


def test_classify_returns_valid_json_structure(client):
    """Successful classification should return a label + confidence in range."""
    response = client.post("/classify", json={"text": "New AI chip announced by startup."})

    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body
    assert isinstance(body["label"], str)
    assert 0.0 <= body["confidence"] <= 1.0


def test_classify_rejects_missing_text_field(client):
    """Missing the required `text` field should trigger a 400, not a 500."""
    response = client.post("/classify", json={})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
