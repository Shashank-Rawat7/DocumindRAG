from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from app.api import health, documents, auth
from app.core.database import Base, engine
from app.core.middleware import RequestLoggingMiddleware
from app.models import document, user

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DocuMind API")

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(auth.router, prefix="/api/v1", tags=["auth"])
app.include_router(documents.router, prefix="/api/v1", tags=["documents"])