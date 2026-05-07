import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

from f1_predictor.common.config import settings

from f1_predictor.features import features

logger = logging.getLogger(__name__)

router = APIRouter()

class RacePredictionRequest(BaseModel):
    year: int
    round: int
    
@router.post("/predict")
async def predict(request: Request, payload: RacePredictionRequest):
    race_data = request.app.state.f1_data
    model = request.app.state.model    
    
    target_df = race_data[(race_data["year"] == payload.year) & (race_data["round"] == payload.round)]

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