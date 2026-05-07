import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from f1_predictor.serve.routes.health import router as health_router
from f1_predictor.serve.routes.predict import router as predict_router
from f1_predictor.common.config import settings
from f1_predictor.serve.startup import load_and_prepare_data, load_inference_model

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model, model_version, model_id = load_inference_model(tag="champion")
    app.state.f1_data, app.state.all_race_data = load_and_prepare_data()
    
    app.state.model = model
    app.state.model_version = model_version
    app.state.model_id = model_id
    yield
    
app = FastAPI(
    title="F1 Predictor",
    description="",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(predict_router)

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