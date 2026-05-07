import pandas as pd

import f1_predictor.data.clean as clean
import  f1_predictor.data.validate as validate
from f1_predictor.data.merge import build_race_frame
from f1_predictor.common.config import settings

import f1_predictor.features.constructor as constructor_features
import f1_predictor.features.context as context_features
import f1_predictor.features.driver as driver_features
from f1_predictor.serve.clients import LakeFSClient, MLFlowClient

def load_and_prepare_data() -> pd.DataFrame:
    lakefs_client = LakeFSClient()

    all_race_data = lakefs_client.load_races()
    
    raw_df = build_race_frame(
        races=all_race_data,
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

    return cleaned_df.sort_values(by=["year", "round"]), all_race_data

def load_inference_model(tag: str):
    return MLFlowClient().get_model(settings.mlflow_experiment_name, tag=tag)
