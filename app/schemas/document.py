from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional


class DocumentCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    filename: str = Field(min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)


class DocumentUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    filename: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)


class DocumentResponse(BaseModel):
    id: int
    filename: str
    description: Optional[str]
    content_type: Optional[str] = None
    size_bytes: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True