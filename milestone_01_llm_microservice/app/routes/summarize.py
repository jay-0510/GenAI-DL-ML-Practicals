"""
summarize.py
------------
Route handler for POST /summarize. Same thin-controller pattern as classify.py:
validation via Pydantic, business logic via BedrockService, route just wires
them together and computes the length metrics the README's response contract
promises.
"""

from fastapi import APIRouter, Depends

from app.models.requests import SummarizeRequest
from app.models.schemas import SummarizeResponse
from app.services.bedrock_service import BedrockService, get_bedrock_service

router = APIRouter(tags=["summarize"])


@router.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize long-form text",
)
async def summarize_text(
    payload: SummarizeRequest,
    service: BedrockService = Depends(get_bedrock_service),
) -> SummarizeResponse:
    """Summarizes `payload.text` and reports original vs. summary word counts."""
    summary = await service.summarize(payload.text, payload.max_length)
    return SummarizeResponse(
        summary=summary,
        original_length=len(payload.text.split()),
        summary_length=len(summary.split()),
    )
