"""
classify.py
-----------
Route handler for POST /classify. Deliberately thin: validation is delegated
to Pydantic (ClassifyRequest), business logic to BedrockService. The route's
only job is wiring the two together and shaping the HTTP response.
"""

from fastapi import APIRouter, Depends

from app.models.requests import ClassifyRequest
from app.models.schemas import ClassifyResponse
from app.services.bedrock_service import BedrockService, get_bedrock_service

router = APIRouter(tags=["classify"])


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify text into a predefined category",
)
async def classify_text(
    payload: ClassifyRequest,
    service: BedrockService = Depends(get_bedrock_service),
) -> ClassifyResponse:
    """Classifies `payload.text` and returns the predicted label + confidence."""
    label, confidence = await service.classify(payload.text)
    return ClassifyResponse(label=label, confidence=confidence)
