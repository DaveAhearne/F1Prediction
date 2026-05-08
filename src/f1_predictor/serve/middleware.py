import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from f1_predictor.serve.log import request_id_ctx

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        token = request_id_ctx.set(str(uuid.uuid4()))
        try:
            return await call_next(request)
        finally:
            request_id_ctx.reset(token)