import logging
from fastapi import FastAPI
import onnxruntime as ort
from f1_predictor.serve.routes.health import router as health_router
from f1_predictor.common.config import settings

logger = logging.getLogger(__name__)
    
app = FastAPI(
    title="F1 Predictor",
    description="",
    version="1.0.0",
    docs_url="/",
)

app.include_router(health_router)

def main() -> None:
    import os
    import uvicorn

    host = os.getenv("HOST", settings.host)
    port = int(os.getenv("PORT", settings.port))
    workers = int(os.getenv("WORKERS", settings.workers))

    uvicorn.run(
        "f1_predictor.serve.api:app",
        host=host,
        port=port,
        workers=workers,
    )