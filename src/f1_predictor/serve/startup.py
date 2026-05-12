import pandas as pd

import  f1_predictor.data.validate as validate
from f1_predictor.common.config import settings

from f1_predictor.pipelines.prepare import prepare_data
from f1_predictor.serve.clients import LakeFSClient, MLFlowClient

def load_and_prepare_data() -> pd.DataFrame:
    lakefs_client = LakeFSClient()

    all_race_data = lakefs_client.load_races()
    all_circuit_data = lakefs_client.load_circuits()

    df = prepare_data(min_year=1990, max_year=None)
    validate.check_schema(df)

    return df.sort_values(by=["year", "round"]), all_race_data, all_circuit_data

def load_inference_model(tag: str):
    return MLFlowClient().get_model(settings.mlflow_experiment_name, tag=tag)
