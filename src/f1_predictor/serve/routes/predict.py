import logging
from fastapi import APIRouter, Request
from pydantic import BaseModel
import onnxruntime as rt
import numpy as np

import f1_predictor.data.clean as clean
import  f1_predictor.data.validate as validate
from f1_predictor.data.merge import build_race_frame
from f1_predictor.common.config import settings

import f1_predictor.features.constructor as constructor_features
import f1_predictor.features.context as context_features
import f1_predictor.features.driver as driver_features
from f1_predictor.features import features

logger = logging.getLogger(__name__)

router = APIRouter()

class RacePredictionRequest(BaseModel):
    year: int
    round: int
    
@router.post("/predict")
async def predict(request: Request, payload: RacePredictionRequest):
    lakefs_client = request.app.state.lakefs_client
    mlflow_client = request.app.state.mlflow_client

    raw_df = build_race_frame(
        races=lakefs_client.load_races(),
        circuits=lakefs_client.load_circuits(),
        constructors=lakefs_client.load_constructors(),
        drivers=lakefs_client.load_drivers(),
        results=lakefs_client.load_results(),
        statuses=lakefs_client.load_statuses()
    )

    cleaned_df = clean.clean_data(raw_df, min_year=1990, max_year=None)
    validate.check_schema(cleaned_df)

    cleaned_df = driver_features.add_driver_rolling_podium_rates(cleaned_df)
    cleaned_df = driver_features.add_driver_circuit_podium_rate(cleaned_df)
    cleaned_df = driver_features.add_driver_experience(cleaned_df)
    cleaned_df = driver_features.add_driver_age(cleaned_df)
    cleaned_df = constructor_features.add_constructor_rolling_podium_rates(cleaned_df)
    cleaned_df = constructor_features.add_constructor_dnf_rates(cleaned_df)
    cleaned_df = context_features.add_championship_position(cleaned_df)
    cleaned_df = context_features.add_season_podium_rate(cleaned_df)
    cleaned_df = context_features.add_home_race(cleaned_df)
    cleaned_df = context_features.add_circuit_type(cleaned_df)
    cleaned_df = context_features.add_regulation_era(cleaned_df)
    cleaned_df = context_features.add_grid_size(cleaned_df)

    sorted_df = cleaned_df.sort_values(by=["year", "round"])

    target_df = sorted_df[(sorted_df["year"] == payload.year) & (sorted_df["round"] == payload.round)]

    model, version, run = mlflow_client.get_model(settings.mlflow_experiment_name, tag="champion")

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