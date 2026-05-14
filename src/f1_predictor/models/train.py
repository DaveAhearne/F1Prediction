import pandas as pd
import mlflow
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.metrics import roc_auc_score, brier_score_loss
from mlflow.data.http_dataset_source import HTTPDatasetSource
from f1_predictor.features import features
from datetime import datetime
from f1_predictor.common.config import settings
from f1_predictor.models import fold, export, types

def train(data: pd.DataFrame, commit_sha, lgbm_params) -> types.TrainingResult:
    folds = fold.generate_rolling_window_folds(data, train_window=10, val_window=1)

    X = data[features.MODEL_FEATURES]
    y = data["podiumFinish"]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"lgbm-walk-forward-{timestamp}"
    
    training_parameters = {
        "run_name": run_name,
        "description": f"LightGBM walk-forward training on {data['year'].min()}-{data['year'].max()} data",
        "tags": {
            "model_type": "lightgbm",
            "feature_set": features.MODEL_FEATURES,
            "data_version": f"{data['year'].min()}-{data['year'].max()}",
            "validation_strategy": "walk-forward"
        },
        "commit_sha": commit_sha,
        "model_params": {
            **lgbm_params,
            "objective": "binary",
            "verbose": -1
        }
    }

    run_name, run_id = train_model_for_folds(X, y, data, folds, training_parameters)

    return types.TrainingResult(run_name=run_name, run_id=run_id)

def train_single_fold(dataFrame: pd.DataFrame, runName: str, fold, X, y, train_years, val_years, model_params):
    train_mask = dataFrame["year"].isin(train_years)
    val_mask   = dataFrame["year"].isin(val_years)

    X_train_fold, y_train_fold = X[train_mask], y[train_mask]
    X_val_fold,   y_val_fold   = X[val_mask],   y[val_mask]

    with mlflow.start_run(run_name=f"{runName}-fold-{fold}", nested=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=types.INTEGER_SCHEMA_WARNING)
            train_dataset = mlflow.data.from_pandas(
                dataFrame[train_mask][features.MODEL_FEATURES + ["podiumFinish"]],
                name=f"{runName}-train-fold-{fold}",
                targets="podiumFinish"
            )
            val_dataset = mlflow.data.from_pandas(
                dataFrame[val_mask][features.MODEL_FEATURES + ["podiumFinish"]],
                name=f"{runName}-val-fold-{fold}",
                targets="podiumFinish"
            )
        mlflow.log_input(train_dataset, context="training")
        mlflow.log_input(val_dataset, context="validation")

        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict_proba(X_val_fold)[:, 1]

        fold_auc   = roc_auc_score(y_val_fold, preds)
        fold_brier = brier_score_loss(y_val_fold, preds)

        mlflow.log_param("train_years", f"{train_years[0]}-{train_years[-1]}")
        mlflow.log_param("val_year", val_years[0])
        mlflow.log_metric("val_roc_auc", fold_auc)
        mlflow.log_metric("val_brier_score", fold_brier)

        return fold_auc, fold_brier, preds, y_val_fold

def train_model_for_folds(X, y, dataFrame, folds, config) -> tuple[str, str]:
    run_name = config["run_name"]
    model_params = config["model_params"]
    description = config["description"]
    commit_sha = config["commit_sha"]

    with mlflow.start_run(run_name=run_name, description=description) as parent_run:
        mlflow.set_tags(config["tags"])

        source = HTTPDatasetSource(url=f"lakefs://{settings.lakefs_repo}/main@{commit_sha}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=types.INTEGER_SCHEMA_WARNING)
            dataset = mlflow.data.from_pandas(
                dataFrame,
                source=source,
                name="f1-race-data",
                targets="podiumFinish"
            )
        mlflow.log_input(dataset, context="training")

        fold_aucs, fold_briers = [], []
        all_preds, all_labels = [], []

        for fold, (train_years, val_years) in enumerate(folds):
            fold_auc, fold_brier, preds, y_val_fold = train_single_fold(
                dataFrame,
                run_name,
                fold,
                X,
                y,
                train_years,
                val_years,
                model_params)

            all_preds.extend(preds)
            all_labels.extend(y_val_fold)
            fold_aucs.append(fold_auc)
            fold_briers.append(fold_brier)

        agg_auc = roc_auc_score(all_labels, all_preds)

        mlflow.log_metric("training_race_count", dataFrame["raceId"].nunique())
        mlflow.log_metric("mean_val_roc_auc", np.mean(fold_aucs))
        mlflow.log_metric("mean_val_brier_score", np.mean(fold_briers))
        mlflow.log_metric("agg_val_roc_auc", agg_auc)

        train_model_on_all_data(run_name, X, y, dataFrame, model_params)

        return run_name, parent_run.info.run_id

def train_model_on_all_data(run_name, X, y, dataFrame, model_params):
    final_model = lgb.LGBMClassifier(**model_params)
    final_model.fit(X, y)

    version = export.log_model_artifacts(final_model, X, y, dataFrame, run_name)

    client = mlflow.MlflowClient()
    client.update_model_version(
        name=settings.mlflow_experiment_name,
        version=version.version,
        description=f"LightGBM walk-forward model, ONNX export. Training run: {run_name}"
    )
    client.set_model_version_tag(
        name=settings.mlflow_experiment_name,
        version=version.version,
        key="model_type",
        value="lightgbm"
    )
    client.set_model_version_tag(
        name=settings.mlflow_experiment_name,
        version=version.version,
        key="export_format",
        value="onnx"
    )

    client.set_registered_model_alias(
        name=settings.mlflow_experiment_name,
        alias="champion",
        version=version.version
    )
    print(f"Champion alias promoted to version {version.version}")