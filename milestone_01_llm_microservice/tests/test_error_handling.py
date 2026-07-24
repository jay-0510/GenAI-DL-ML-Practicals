"""
test_error_handling.py
-----------------------
Covers invalid inputs and error-envelope consistency across endpoints.
This satisfies the README's third required test category.
"""


def test_classify_empty_payload_returns_error_envelope(client):
    """Invalid input on /classify should return the documented error shape."""
    response = client.post("/classify", json={"text": ""})

    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert body["error"]["code"] == "VALIDATION_ERROR"


def test_summarize_invalid_max_length_returns_error(client):
    """max_length below the allowed minimum (10) should be rejected as invalid."""
    response = client.post("/summarize", json={"text": "some text", "max_length": 1})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_route_returns_404():
    """Sanity check that non-existent routes don't get swallowed by our catch-all handler."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        response = c.get("/does-not-exist")
    assert response.status_code == 404
