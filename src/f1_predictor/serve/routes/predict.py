import logging
from fastapi import APIRouter, Request

from f1_predictor.features import features
from f1_predictor.serve import prepare
from f1_predictor.serve.inference import ONNXModelInference
from f1_predictor.serve.schema import DriverPrediction, RacePredictionRequest
from f1_predictor.serve.template_env import templates

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/predict")
async def get_predict(request: Request):
    return templates.TemplateResponse(request, "home.html")

@router.post("/predict", response_model=list[DriverPrediction])
async def predict(request: Request, payload: RacePredictionRequest):
    f1_data = request.app.state.f1_data
    all_race_data = request.app.state.all_race_data
    all_circuit_data = request.app.state.all_circuit_data

    model_session = ONNXModelInference(request.app.state.model)

    race_features = prepare.build_race_features(f1_data, all_race_data, all_circuit_data, payload.year, payload.round)
    X = race_features[features.MODEL_FEATURES]

    preds = model_session.predict(X)

    return sorted(
        [
            DriverPrediction(
                driverId=int(row["driverId"]),
                driver=f"{row['forename']} {row['surname']}",
                podium_probability=round(float(prob), 4)
            )
            for (_, row), prob in zip(race_features.iterrows(), preds)
        ],
        key=lambda x: x.podium_probability,
        reverse=True
    )