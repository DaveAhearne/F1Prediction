import warnings
import mlflow
import f1_predictor.data.clean as clean
import  f1_predictor.data.validate as validate
import f1_predictor.data.load as data_loaders
from f1_predictor.data.merge import build_race_frame
import f1_predictor.features.constructor as constructor_features
import f1_predictor.features.context as context_features
import f1_predictor.features.driver as driver_features
from f1_predictor.models.train import train
from f1_predictor.common.config import settings

def train_pipeline():
    warnings.filterwarnings("error", message="Hint: Inferred schema contains integer column")
    
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    
    raw_df = build_race_frame(
        races=data_loaders.load_races(),
        circuits=data_loaders.load_circuits(),
        constructors=data_loaders.load_constructors(),
        drivers=data_loaders.load_drivers(),
        results=data_loaders.load_results(),
        statuses=data_loaders.load_statuses()
    )

    lakefs_commit_sha = data_loaders.get_commit_sha()

    cleaned_df = clean.clean_data(raw_df, min_year=1990, max_year=2025)
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

    result = train(data=cleaned_df, commit_sha=lakefs_commit_sha)
    print(f"\nTraining finished: \n\tRun name: {result.run_name} \n\tRun id: {result.run_id}\n")