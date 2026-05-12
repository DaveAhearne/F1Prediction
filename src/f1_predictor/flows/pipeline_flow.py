from prefect import flow, task
from ingest.update import run as ingest_run
from f1_predictor.pipelines.train_pipeline import train_pipeline
from f1_predictor.common.config import settings

@task
def ingest_new_data():
    ingest_run()

@task
def run_training():
    train_pipeline()

@flow(name=settings.prefect_flow_name, log_prints=True)
def f1_pipeline():
    ingest_new_data()
    run_training()

if __name__ == "__main__":
    f1_pipeline.serve(
        name="scheduled",
        cron=settings.prefect_cron_schedule
    )