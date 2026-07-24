"""
config.py
---------
Centralizes environment configuration using pydantic-settings.
Why: avoids scattering `os.getenv()` calls across the codebase and gives
us validation (e.g. wrong types, missing required vars) at startup instead
of failing deep inside a request handler.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings, populated from environment variables / .env."""

    aws_region: str = "us-east-1"
    bedrock_model_id: str = "amazon.nova-micro-v1:0"

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None

    app_env: str = "development"
    log_level: str = "INFO"

    # Classification taxonomy is configurable rather than hardcoded,
    # since the README doesn't fix one and hardcoding it would silently
    # bias every classification call.
    classification_labels: str = "finance,technology,sports,health,politics,entertainment,other"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def labels_list(self) -> list[str]:
        """Splits the comma-separated label string into a clean list."""
        return [label.strip() for label in self.classification_labels.split(",") if label.strip()]


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor.
    Why lru_cache: Settings() re-reads and re-validates the environment on every
    call; caching means we parse .env once per process, not once per request.
    """
    return Settings()
