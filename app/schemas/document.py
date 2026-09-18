from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class DocumentCreate(BaseModel):
    filename: str
    description: Optional[str] = None


class DocumentUpdate(BaseModel):
    filename: Optional[str] = None
    description: Optional[str] = None


class DocumentResponse(BaseModel):
    id: int
    filename: str
    description: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True