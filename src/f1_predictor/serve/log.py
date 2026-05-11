import logging
from f1_predictor.common.config import settings
from contextvars import ContextVar
import uuid
from starlette.middleware.base import BaseHTTPMiddleware

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        token = request_id_ctx.set(str(uuid.uuid4()))
        try:
            return await call_next(request)
        finally:
            request_id_ctx.reset(token)

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get() or "-"
        return True

def configure_logging() -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | request_id=%(request_id)s | %(message)s"
    )

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level.upper())

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        root_logger.addHandler(handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)