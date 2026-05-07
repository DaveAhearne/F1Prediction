import onnx
import pandas as pd
import mlflow
import numpy as np
import lightgbm as lgb
import warnings
from sklearn.metrics import roc_auc_score, brier_score_loss
from mlflow.models import infer_signature
from mlflow.data.http_dataset_source import HTTPDatasetSource
from f1_predictor.features import features
from datetime import datetime
from f1_predictor.common.config import settings
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

INTEGER_SCHEMA_WARNING = "Hint: Inferred schema contains integer column"

def train(data: pd.DataFrame, commit_sha):
    folds = generate_rolling_window_folds(data, train_window=10, val_window=1)

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
            'n_estimators': settings.train_n_estimators,
            'learning_rate': settings.train_learning_rate,
            'num_leaves': settings.train_num_leaves,
            'min_child_samples': settings.train_min_child_samples,
            'scale_pos_weight': settings.train_scale_pos_weight,
            'subsample': settings.train_subsample,
            'colsample_bytree': settings.train_colsample_bytree,
            'objective': 'binary',
            'verbose': -1
        }
    }

    run_name, run_id = train_model_for_folds(X, y, data, folds, training_parameters)

    return run_name, run_id

def generate_rolling_window_folds(dataframe, train_window, val_window):
    years = sorted(dataframe["year"].unique())

    folds = []

    for i in range(len(years) - train_window - val_window + 1):
        train_years = years[i: i + train_window]
        val_years = years[i + train_window: i + train_window + val_window]
        folds.append((train_years, val_years))

    return folds

def train_single_fold(dataFrame: pd.DataFrame, runName: str, fold, X, y, train_years, val_years, model_params):
    train_mask = dataFrame["year"].isin(train_years)
    val_mask   = dataFrame["year"].isin(val_years)

    X_train_fold, y_train_fold = X[train_mask], y[train_mask]
    X_val_fold,   y_val_fold   = X[val_mask],   y[val_mask]

    with mlflow.start_run(run_name=f"{runName}-fold-{fold}", nested=True):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=INTEGER_SCHEMA_WARNING)
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

        return model, fold_auc, fold_brier, preds, y_val_fold

def train_model_for_folds(X, y, dataFrame, folds, config) -> tuple[str, str]:
    run_name = config["run_name"]
    model_params = config["model_params"]
    description = config["description"]
    commit_sha = config["commit_sha"]

    with mlflow.start_run(run_name=run_name, description=description) as parent_run:
        mlflow.set_tags(config["tags"])

        source = HTTPDatasetSource(url=f"lakefs://{settings.lakefs_repo}/main@{commit_sha}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=INTEGER_SCHEMA_WARNING)
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
            model, fold_auc, fold_brier, preds, y_val_fold = train_single_fold(
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

        mlflow.log_metric("mean_val_roc_auc", np.mean(fold_aucs))
        mlflow.log_metric("mean_val_brier_score", np.mean(fold_briers))
        mlflow.log_metric("agg_val_roc_auc", agg_auc)

        train_model_on_all_data(run_name, X, y, model_params)

        return run_name, parent_run.info.run_id

def convert_to_onnx(model: lgb.LGBMClassifier) -> onnx.ModelProto:
    initial_types = [("input", FloatTensorType([None, len(model.booster_.feature_name())]))]
    onnx_model = onnxmltools.convert_lightgbm(model, initial_types=initial_types, target_opset=12)

    return onnx_model

def log_model_artifact(model, X, run_name):
    mlflow.log_params(model.get_params())

    output_sample = pd.DataFrame(
        model.predict_proba(X)[:, 1],
        columns=["podium_probability"]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=INTEGER_SCHEMA_WARNING)
        signature = infer_signature(X, output_sample)
        mlflow.lightgbm.log_model(model, name=run_name, signature=signature)

def train_model_on_all_data(run_name, X, y, model_params):
    final_model = lgb.LGBMClassifier(**model_params)
    final_model.fit(X, y)

    log_model_artifact(final_model, X, run_name)

    # TODO: Move this out of here, we should really only do this on promoted runs
    # but, until the training loop has been automated this is a good enough fit for now
    onnx_model = convert_to_onnx(final_model)

    mlflow.onnx.log_model(onnx_model, name=f"{run_name}-onnx")