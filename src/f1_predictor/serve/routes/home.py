import logging
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def home(request: Request):
    return RedirectResponse(url="/predict", status_code=303)