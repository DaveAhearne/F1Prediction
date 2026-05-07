import warnings
import mlflow
import onnxruntime as rt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from f1_predictor.common.config import settings
from f1_predictor.models import types
from mlflow.data.http_dataset_source import HTTPDatasetSource

def get_model_from_mlflow(experiment_name, tag):
    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(experiment_name, tag)
    model = mlflow.onnx.load_model(f"models:/{experiment_name}@{tag}")
    return model, model_version.version, model_version.run_id

def get_roc_auc_stats(all_preds, all_labels):
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    auc   = roc_auc_score(all_labels, all_preds)
    brier = brier_score_loss(all_labels, all_preds)
    return (fpr, tpr, auc, brier)

def run_inference(model, X):
    sess = rt.InferenceSession(model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    preds = sess.run(None, {input_name: X.values.astype(np.float32)})[1][:,1]

    return preds

def evaluate(X, y, data_sha, tag="champion") -> types.EvaluationResult:
    experiment_name = settings.mlflow_experiment_name

    model, model_version, model_run_id = get_model_from_mlflow(experiment_name, tag)

    preds = run_inference(model, X)

    fpr, tpr, auc, brier = get_roc_auc_stats(preds, y)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name=f"evaluation-{tag}-{timestamp}"):
        mlflow.set_tag("model_name", experiment_name)
        mlflow.set_tag("model_alias", tag)
        mlflow.set_tag("model_version", model_version)
        mlflow.set_tag("model_run_id", model_run_id)
        mlflow.set_tag("evaluation_type", "test_set")
        mlflow.log_metric("test_roc_auc", auc)
        mlflow.log_metric("test_brier_score", brier)

        source = HTTPDatasetSource(url=f"lakefs://{settings.lakefs_repo}/main@{data_sha}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=types.INTEGER_SCHEMA_WARNING)
            dataset = mlflow.data.from_pandas(
                pd.concat([X, y], axis=1),
                source=source,
                name="f1-race-data",
                targets="podiumFinish"
            )
        mlflow.log_input(dataset, context="evaluation")

    return types.EvaluationResult(auc=auc, brier=brier, fpr=fpr, tpr=tpr)