import onnx
import warnings
import onnxmltools
import mlflow
import lightgbm as lgb
import pandas as pd
import tempfile
import os
from mlflow.models import infer_signature
from onnxmltools.convert.common.data_types import FloatTensorType
from f1_predictor.models import types
from f1_predictor.common.config import settings

def convert_to_onnx(model: lgb.LGBMClassifier) -> onnx.ModelProto:
    initial_types = [("input", FloatTensorType([None, len(model.booster_.feature_name())]))]
    onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types, target_opset=12)

    return onnx_model

def log_model_artifacts(model, X, y, data, run_name, last_n_races=72):
    mlflow.log_params(model.get_params())
    
    last_n_race_ids = (
        data[["raceId", "year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .tail(last_n_races)["raceId"]
    )
    recent_mask = data["raceId"].isin(last_n_race_ids)
    reference = X[recent_mask].copy()
    reference["podiumFinish"] = y[recent_mask].values

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, "training_reference.parquet")
        reference.to_parquet(tmp_path, index=False)
        mlflow.log_artifact(tmp_path)

    output_sample = pd.DataFrame(
        model.predict_proba(X)[:, 1],
        columns=["podium_probability"]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=types.INTEGER_SCHEMA_WARNING)
        signature = infer_signature(X, output_sample)
        mlflow.lightgbm.log_model(model, name=run_name, signature=signature)

        onnx_model = convert_to_onnx(model)
        mlflow.onnx.log_model(
            onnx_model,
            name=f"{run_name}-onnx",
            signature=signature,
            registered_model_name=settings.mlflow_experiment_name
        )

        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(settings.mlflow_experiment_name)
        return versions[-1]