from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class DocumentCreate(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)


class DocumentUpdate(BaseModel):
    filename: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1000)


class DocumentResponse(BaseModel):
    id: int
    filename: str
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True