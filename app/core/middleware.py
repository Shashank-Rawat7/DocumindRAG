import time
import uuid
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


logger = logging.getLogger("documind")


class RequestLoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())

        request.state.request_id = request_id

        start_time = time.perf_counter()

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            f"request_id={request_id} "
            f"method={request.method} "
            f"path={request.url.path} "
            f"status_code={response.status_code} "
            f"duration_ms={duration_ms}"
        )

        response.headers["X-Request-ID"] = request_id

        return response