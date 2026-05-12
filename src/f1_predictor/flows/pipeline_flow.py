import tempfile
import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace
from prefect import flow, task
from ingest.update import run as ingest_run
from f1_predictor.pipelines.prepare import prepare_data as _prepare_data
from f1_predictor.pipelines.train_pipeline import train_pipeline
from f1_predictor.features.features import MODEL_FEATURES
from f1_predictor.common.config import settings
from evidently.pipeline.column_mapping import ColumnMapping

@task
def ingest_new_data():
    ingest_run()

@task
def build_current_frame() -> pd.DataFrame:
    return _prepare_data(min_year=1990, max_year=2025)

@task
def check_drift(current_df: pd.DataFrame) -> bool:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.MlflowClient()

    model_version = client.get_model_version_by_alias(
        settings.mlflow_experiment_name, "champion"
    )
    run_id = model_version.run_id

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, "training_reference.parquet", tmp_dir)
        reference_df = pd.read_parquet(local_path)

    column_mapping = ColumnMapping()
    column_mapping.categorical_features = ["driverId", "constructorId", "circuitId", "regulation_era", "is_home_race"]
    column_mapping.numerical_features = [
        f for f in MODEL_FEATURES 
        if f not in column_mapping.categorical_features
    ]

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_df[MODEL_FEATURES],
        current_data=current_df[MODEL_FEATURES],
        column_mapping=column_mapping
    )

    workspace = RemoteWorkspace(settings.evidently_workspace_url)
    workspace.add_report(settings.evidently_project_id, report)

    result = report.as_dict()
    return result["metrics"][0]["result"]["dataset_drift"]

@task
def run_training():
    train_pipeline()

@flow(name=settings.prefect_flow_name, log_prints=True)
def f1_pipeline():
    ingest_new_data()
    current_df = build_current_frame()

    should_retrain = check_drift(current_df)
    if should_retrain:
        run_training()

if __name__ == "__main__":
    f1_pipeline.serve(
        name="scheduled",
        cron=settings.prefect_cron_schedule
    )