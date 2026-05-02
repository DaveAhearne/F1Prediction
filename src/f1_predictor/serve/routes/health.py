import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health():
    logger.info("HIT: /health endpoint")
    return {"status": "ok"}