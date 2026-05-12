import warnings
import mlflow
import  f1_predictor.data.validate as validate
import f1_predictor.data.load as data_loaders
from f1_predictor.features import features
from f1_predictor.models.eval import evaluate
from f1_predictor.common.config import settings
from f1_predictor.pipelines.prepare import prepare_data

def eval_pipeline():
    warnings.filterwarnings("error", message="Hint: Inferred schema contains integer column")
    
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    df = prepare_data(min_year=1990, max_year=2026)
    validate.check_schema(df)

    eval_df = df[df["year"] == 2025]

    X = eval_df[features.MODEL_FEATURES]
    y = eval_df["podiumFinish"]

    lakefs_commit_sha = data_loaders.get_commit_sha()
    evaluate(X, y, lakefs_commit_sha, tag="champion")