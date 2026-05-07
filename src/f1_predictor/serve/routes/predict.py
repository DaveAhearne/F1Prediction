import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

from f1_predictor.features import features
from f1_predictor.serve.request import get_target_df
from f1_predictor.serve.template_env import templates

logger = logging.getLogger(__name__)

router = APIRouter()

class RacePredictionRequest(BaseModel):
    year: int
    round: int

@router.get("/")
async def home(request: Request):
    return RedirectResponse(url="/predict", status_code=303)

@router.get("/predict")
async def get_predict(request: Request):
    return templates.TemplateResponse(request, "home.html")

@router.post("/predict")
async def predict(request: Request, payload: RacePredictionRequest):
    f1_data = request.app.state.f1_data
    all_race_data = request.app.state.all_race_data
    model = request.app.state.model

    target_df = get_target_df(f1_data, all_race_data, payload.year, payload.round)

    X = target_df[features.MODEL_FEATURES]

    sess = rt.InferenceSession(model.SerializeToString())

    input_name = sess.get_inputs()[0].name

    outputs = sess.run(None, {input_name: X.values.astype(np.float32)})
    preds = np.array([d[1] for d in outputs[1]])

    results = []
    for (_, row), prob in zip(target_df.iterrows(), preds):
        results.append({
            "driverId": int(row["driverId"]),
            "driver": f"{row['forename']} {row['surname']}",
            "podium_probability": round(float(prob), 4)
        })

    results = sorted(results, key=lambda x: x["podium_probability"], reverse=True)
    return results