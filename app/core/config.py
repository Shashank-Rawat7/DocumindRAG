from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()