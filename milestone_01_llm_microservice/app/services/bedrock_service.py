"""
bedrock_service.py
-------------------
All AWS Bedrock interaction lives here: client setup, prompt construction,
invocation, and response parsing. Routes never touch boto3 directly — if we
swap providers later (e.g. Bedrock -> another model host), only this
file changes.

Uses the Amazon Bedrock Converse API so the same code works with
Amazon Nova, Claude, Llama, Mistral, and other supported models.
"""

import asyncio
import json
import logging

import boto3
from botocore.exceptions import (
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)

from app.config import get_settings
from app.utils.exceptions import AuthenticationError, BedrockError

logger = logging.getLogger("llm_microservice")


class BedrockService:
    """Thin wrapper around the Bedrock Runtime client."""

    def __init__(self) -> None:
        settings = get_settings()

        self.model_id = settings.bedrock_model_id
        self.labels = settings.labels_list

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        )

    def _invoke_sync(self, prompt: str, max_tokens: int) -> str:
        """
        Blocking Bedrock Converse API call.
        """

        try:
            response = self._client.converse(
                modelId=self.model_id,
                system=[
                    {
                        "text": (
                            "You are a helpful AI assistant. "
                            "Follow the user's instructions exactly. "
                            "Return only the requested output."
                        )
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt
                            }
                        ],
                    }
                ],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": 0.3,
                },
            )

        except NoCredentialsError as exc:
            raise AuthenticationError(details=str(exc)) from exc

        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")

            if error_code in (
                "UnrecognizedClientException",
                "AccessDeniedException",
            ):
                raise AuthenticationError(details=str(exc)) from exc

            raise BedrockError(details=str(exc)) from exc

        except EndpointConnectionError as exc:
            raise BedrockError(
                message="Could not reach AWS Bedrock endpoint",
                details=str(exc),
            ) from exc

        try:
            return response["output"]["message"]["content"][0]["text"]

        except (KeyError, IndexError) as exc:
            raise BedrockError(
                message="Unexpected response shape from Bedrock",
                details=str(exc),
            ) from exc

    async def _invoke(self, prompt: str, max_tokens: int) -> str:
        """
        Async wrapper around the synchronous boto3 client.
        """
        return await asyncio.to_thread(
            self._invoke_sync,
            prompt,
            max_tokens,
        )

    async def classify(self, text: str) -> tuple[str, float]:
        """
        Classify text into one of the configured labels.
        """

        labels_joined = ", ".join(self.labels)

        prompt = (
            f"Classify the following text into exactly one of these categories: "
            f"{labels_joined}.\n\n"
            f'Respond ONLY with valid JSON in this format:\n'
            f'{{"label":"<category>","confidence":0.95}}\n\n'
            f"Text:\n{text}"
        )

        raw = await self._invoke(prompt, max_tokens=100)

        try:
            parsed = json.loads(raw.strip())

            label = str(parsed["label"])
            confidence = float(parsed["confidence"])

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise BedrockError(
                message="Could not parse classification result",
                details=str(exc),
            ) from exc

        if label not in self.labels:
            logger.warning(
                "Model returned unexpected label '%s'",
                label,
            )

            label = (
                "other"
                if "other" in self.labels
                else self.labels[-1]
            )

        confidence = max(0.0, min(confidence, 1.0))

        return label, confidence

    async def summarize(
        self,
        text: str,
        max_length: int,
    ) -> str:
        """
        Summarize long text.
        """

        prompt = (
            f"Summarize the following text in no more than "
            f"{max_length} words.\n\n"
            f"Return ONLY the summary.\n\n"
            f"Text:\n{text}"
        )

        summary = await self._invoke(
            prompt,
            max_tokens=max_length * 2 + 50,
        )

        return summary.strip()


_service_instance: BedrockService | None = None


def get_bedrock_service() -> BedrockService:
    """
    Returns a singleton Bedrock service instance.
    """

    global _service_instance

    if _service_instance is None:
        _service_instance = BedrockService()

    return _service_instance