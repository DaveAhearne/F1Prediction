import tempfile
import pandas as pd
import mlflow
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

DRIFT_EXCLUDE = ["year", "round", "driver_experience"]
DRIFT_FEATURES = [f for f in MODEL_FEATURES if f not in DRIFT_EXCLUDE]
REPORT_FEATURES = DRIFT_FEATURES + ["podiumFinish"]

@task
def ingest_new_data():
    ingest_run()

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
    column_mapping.categorical_features = ["driverId", "constructorId", "circuitId", "regulation_era", "is_home_race"]
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
    workspace.add_report(settings.evidently_project_id, report)

    result = report.as_dict()
    return result["metrics"][0]["result"]["dataset_drift"]

@task
def should_force_retrain(full_df: pd.DataFrame) -> bool:
    """
    Force retraining if 12 or more races have been ingested since the
    champion model was last trained, regardless of drift detection.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(
        settings.mlflow_experiment_name, "champion"
    )
    run = client.get_run(model_version.run_id)
    training_race_count = int(run.data.metrics["training_race_count"])
    current_race_count = full_df["raceId"].nunique()
    return (current_race_count - training_race_count) >= 12

@task
def run_training():
    train_pipeline()

@flow(name=settings.prefect_flow_name, log_prints=True)
def f1_pipeline():
    ingest_new_data()
    full_df = build_full_frame()
    current_df = build_current_frame(full_df)
    reference_df = get_reference_frame_from_champion()
    should_retrain = check_drift(current_df, reference_df) or should_force_retrain(full_df)
    if should_retrain:
        run_training()

if __name__ == "__main__":
    f1_pipeline.serve(
        name="scheduled",
        cron=settings.prefect_cron_schedule
    )