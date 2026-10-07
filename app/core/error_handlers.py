import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("documind")


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        f"Unhandled error | "
        f"request_id={request_id} | "
        f"path={request.url.path} | "
        f"method={request.method} | "
        f"error={repr(exc)}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )