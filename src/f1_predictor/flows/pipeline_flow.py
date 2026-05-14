import json
import tempfile
import pandas as pd
import mlflow
import asyncio
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.ui.workspace import RemoteWorkspace
from prefect import flow, task
from ingest.update import run as ingest_run
from f1_predictor.pipelines.prepare import prepare_data as _prepare_data
from f1_predictor.pipelines.train_pipeline import train_pipeline
from f1_predictor.features.features import MODEL_FEATURES
from f1_predictor.common.config import settings
from evidently.pipeline.column_mapping import ColumnMapping
from prefect.variables import Variable

DRIFT_EXCLUDE = ["year", "round", "driver_experience", "grid_size", "regulation_era"]
DRIFT_FEATURES = [f for f in MODEL_FEATURES if f not in DRIFT_EXCLUDE]
REPORT_FEATURES = DRIFT_FEATURES + ["podiumFinish"]
ALL_CATEGORICALS = ["driverId", "constructorId", "circuitId", "regulation_era", "is_home_race"]

def parse_variable(value):
    return value if isinstance(value, dict) else json.loads(value)

def get_or_create_project(workspace: RemoteWorkspace, name: str):
    for project in workspace.list_projects():
        if project.name == name:
            return project
    return workspace.create_project(name)

@task
def ingest_new_data() -> bool:
    return ingest_run()

@task
def build_full_frame() -> pd.DataFrame:
    """
    Load the full prepared dataset from LakeFS with no year ceiling.
    """
    return _prepare_data(min_year=1990, max_year=None)

@task
def build_current_frame(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Slice the last 12 races from the full frame to use as the current
    distribution for drift detection. Uses raceId to avoid assumptions
    about race cadence or calendar.
    """
    last_12_race_ids = (
        full_df[["raceId", "year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
        .tail(12)["raceId"]
    )
    return full_df[full_df["raceId"].isin(last_12_race_ids)]

@task
def get_reference_frame_from_champion() -> pd.DataFrame:
    """
    Fetch the training reference parquet logged against the champion model
    run in MLflow. This is the distribution the model was trained on.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(
        settings.mlflow_experiment_name, "champion"
    )
    run_id = model_version.run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, "training_reference.parquet", tmp_dir)
        return pd.read_parquet(local_path)


@task
def check_drift(current_df: pd.DataFrame, reference_df: pd.DataFrame) -> bool:
    column_mapping = ColumnMapping()
    column_mapping.target = "podiumFinish"
    column_mapping.categorical_features = [f for f in ALL_CATEGORICALS if f in DRIFT_FEATURES]
    column_mapping.numerical_features = [
        f for f in DRIFT_FEATURES
        if f not in column_mapping.categorical_features
    ]

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
        TargetDriftPreset(),
    ])
    report.run(
        reference_data=reference_df[REPORT_FEATURES],
        current_data=current_df[REPORT_FEATURES],
        column_mapping=column_mapping
    )

    workspace = RemoteWorkspace(settings.evidently_workspace_url)
    project = get_or_create_project(workspace, settings.evidently_project_name)
    workspace.add_report(project.id, report)

    result = report.as_dict()
    return result["metrics"][0]["result"]["dataset_drift"]

@task
async def should_force_retrain(full_df: pd.DataFrame) -> bool:
    """
    Force retraining if N or more races have been ingested since the
    champion model was last trained, regardless of drift detection.
    """
    retraining_config = parse_variable(await Variable.get("retraining_config"))

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(
        settings.mlflow_experiment_name, "champion"
    )
    run = client.get_run(model_version.run_id)
    training_race_count = int(run.data.metrics["training_race_count"])
    current_race_count = full_df["raceId"].nunique()
    return (current_race_count - training_race_count) >= retraining_config["race_threshold"] 

@task
async def run_training(reason: str):
    lgbm_params = parse_variable(await Variable.get("lgbm_hyperparameters"))
    print(f"Retraining triggered: {reason} - {lgbm_params}")
    train_pipeline(lgbm_params)

@task
def champion_exists() -> bool:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    try:
        client.get_model_version_by_alias(settings.mlflow_experiment_name, "champion")
        return True
    except mlflow.exceptions.MlflowException:
        return False

@flow(name=settings.prefect_flow_name, log_prints=True)
async def f1_pipeline():
    if not champion_exists():
        print("No model exists — running first train")
        await run_training("no champion model exists")
        return

    has_new_data = ingest_new_data()
    if not has_new_data:
        print("No new race data — skipping drift check and training")
        return

    full_df = build_full_frame()
    current_df = build_current_frame(full_df)
    reference_df = get_reference_frame_from_champion()

    drift_detected = check_drift(current_df, reference_df)
    force_retrain = await should_force_retrain(full_df)

    if drift_detected and force_retrain:
        await run_training("drift detected and 12-race threshold reached")
    elif drift_detected:
        await run_training("drift detected")
    elif force_retrain:
        await run_training("12-race threshold reached")
    else:
        print("No retraining required")

if __name__ == "__main__":
    asyncio.run(f1_pipeline.serve(
        name="scheduled",
        cron=settings.prefect_cron_schedule
    ))