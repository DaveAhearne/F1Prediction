import logging
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def home(request: Request):
    logger.info("HIT: / - redirecting to /predict")
    return RedirectResponse(url="/predict", status_code=303)