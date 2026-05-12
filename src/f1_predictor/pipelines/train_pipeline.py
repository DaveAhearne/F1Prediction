import warnings
import mlflow
import  f1_predictor.data.validate as validate
import f1_predictor.data.load as data_loaders
from f1_predictor.models.train import train
from f1_predictor.common.config import settings
from f1_predictor.pipelines.prepare import prepare_data

def train_pipeline():
    warnings.filterwarnings("error", message="Hint: Inferred schema contains integer column")
    
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = prepare_data(min_year=1990, max_year=2025)
    validate.check_schema(df)

    lakefs_commit_sha = data_loaders.get_commit_sha()

    result = train(data=df, commit_sha=lakefs_commit_sha)
    print(f"\nTraining finished: \n\tRun name: {result.run_name} \n\tRun id: {result.run_id}\n")