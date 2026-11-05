import re

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_COMMIT_HASH = re.compile(r"[0-9a-f]{40}")


class Settings(BaseSettings):
    app_name: str = "DocuMind API"
    environment: str = "development"

    database_url: str
    secret_key: str = Field(min_length=32)
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # File upload settings
    upload_dir: str = "uploads"
    max_upload_mb: int = 10
    allowed_extensions: list[str] = [".pdf", ".txt", ".md", ".docx"]

    # Embedding settings (C2.4). The model is loaded from the local Hugging Face
    # cache only, at this exact commit; nothing is downloaded at runtime.
    embedding_model_id: str = Field(default="intfloat/multilingual-e5-small", min_length=1)
    embedding_model_revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    # Inputs per forward pass. 1 was chosen from the C2.3d CPU benchmark
    # (within 10% of the best English and Hindi throughput, no padding waste).
    embedding_batch_size: int = Field(default=1, ge=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    @field_validator("embedding_model_id")
    @classmethod
    def _model_id_is_exact(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError(
                "embedding_model_id must be non-empty and have no leading or trailing whitespace"
            )
        return value

    @field_validator("embedding_model_revision")
    @classmethod
    def _revision_is_commit_hash(cls, value: str) -> str:
        if not _COMMIT_HASH.fullmatch(value):
            raise ValueError("embedding_model_revision must be a 40-character lowercase hex commit hash")
        return value


settings = Settings()
