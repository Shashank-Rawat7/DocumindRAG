from fastapi import FastAPI
from app.api import health, documents
from app.core.database import Base, engine
from app.models import document

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DocuMind API")

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])