import pandas as pd

def train(data: pd.DataFrame):
    folds = GenerateFolds(data,train_window=10, val_window=1)

    engineered_features = [
        "driverId", "constructorId", "circuitId", "year", "round",
        "driver_podium_rate_3", "driver_podium_rate_5", "driver_podium_rate_10",
        "constructor_podium_rate_3", "constructor_podium_rate_5", "constructor_podium_rate_10",
        "constructor_mechanical_dnf_rate_5", "driverAge", "driver_experience", "driver_circuit_podium_rate",
        "driver_championship_position", "driver_season_podium_rate", "grid_size", "regulation_era", "is_home_race"
    ]

    engineered_cat_features = ["driverId", "constructorId", "circuitId", "regulation_era", "is_home_race"]

    X, y = GenXy(data, engineered_features, engineered_cat_features)

    best_params = study.best_params
    best_params["objective"] = "binary"
    best_params["verbose"] = -1

    model, all_preds, all_labels = TrainModel(X, y, final_f1_feature_df, folds, {
        "run_name": "lgbm-tuned-walk-forward",
        "description": "LightGBM with full feature set and Bayesian tuned hyperparameters",
        "tags": {
            "model_type": "lightgbm",
            "feature_set": "full",
            "data_version": "1990-2024",
            "validation_strategy": "walk-forward"
        },
        "model_params": best_params
    })

def GenXy(dataFrame, engineered_features, engineered_cat_features):
    """
    Prepares feature matrix X and target vector y from a DataFrame.
    Casts categorical columns to the 'category' dtype required by LightGBM.

    Args:
        dataFrame: DataFrame containing all features and the 'podiumFinish' target column.
        engineered_features: List of column names to use as model features.
        engineered_cat_features: Subset of engineered_features to cast to 'category' dtype.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the binary target series.
    """
    for col in engineered_cat_features:
        dataFrame[col] = dataFrame[col].astype("category")
    
    X = dataFrame[engineered_features]
    y = dataFrame["podiumFinish"]
    
    return (X,y)

def GenerateFolds(dataframe, train_window, val_window):
	"""
    Generates walk-forward cross-validation folds based on season years.

    Args:
        dataframe: DataFrame containing a 'year' column used to derive the fold splits.
        train_window: Number of seasons to include in each training fold.
        val_window: Number of seasons to include in each validation fold.

    Returns:
        List of (train_years, val_years) tuples, one per fold.
    """
	years = sorted(dataframe["year"].unique())

	folds = []

	for i in range(len(years) - train_window - val_window + 1):
		train_years = years[i: i + train_window]
		val_years = years[i + train_window: i + train_window + val_window]
		folds.append((train_years, val_years))
	    
	return folds

def TrainModel(X, y, dataFrame, folds, config):
    """
    Runs walk-forward cross-validation and logs results to MLflow.
    Logs per-fold metrics to nested runs, and aggregate metrics, model params,
    ROC curve, and the final fold model artifact to the parent run.
    Args:
        X: Feature matrix, aligned with dataFrame's index.
        y: Binary target series (podiumFinish), aligned with dataFrame's index.
        dataFrame: Full DataFrame used for MLflow dataset logging and year-based fold masking.
        folds: List of (train_years, val_years) tuples as returned by GenerateFolds.
        config: Dict containing 'run_name', 'description', 'tags', and 'model_params'.
    Returns:
        Tuple of (model, all_preds, all_labels) where model is the trained LGBMClassifier
        from the final fold, all_preds are the predicted probabilities accumulated across
        all folds, and all_labels are the corresponding true binary labels.
    """
    from mlflow.models import infer_signature

    with mlflow.start_run(run_name=config["run_name"], description=config["description"]) as parent_run:
        mlflow.set_tags(config["tags"])

        fold_aucs, fold_briers = [], []
        all_preds, all_labels = [], []

        for fold, (train_years, val_years) in enumerate(folds):
            train_mask = dataFrame["year"].isin(train_years)
            val_mask   = dataFrame["year"].isin(val_years)

            X_train_fold, y_train_fold = X[train_mask], y[train_mask]
            X_val_fold,   y_val_fold   = X[val_mask],   y[val_mask]

            with mlflow.start_run(run_name=f"lgbm-baseline-fold-{fold}", nested=True):
                train_dataset = mlflow.data.from_pandas(
                    dataFrame[train_mask],
                    name=f"{config['run_name']}-train-fold-{fold}",
                    targets="podiumFinish"
                )
                val_dataset = mlflow.data.from_pandas(
                    dataFrame[val_mask],
                    name=f"{config['run_name']}-val-fold-{fold}",
                    targets="podiumFinish"
                )
                mlflow.log_input(train_dataset, context="training")
                mlflow.log_input(val_dataset, context="validation")

                model = lgb.LGBMClassifier(**config["model_params"])
                model.fit(X_train_fold, y_train_fold)
                preds = model.predict_proba(X_val_fold)[:, 1]

                all_preds.extend(preds)
                all_labels.extend(y_val_fold)

                fold_auc   = roc_auc_score(y_val_fold, preds)
                fold_brier = brier_score_loss(y_val_fold, preds)
                fold_aucs.append(fold_auc)
                fold_briers.append(fold_brier)

                mlflow.log_param("train_years", f"{train_years[0]}-{train_years[-1]}")
                mlflow.log_param("val_year", val_years[0])
                mlflow.log_metric("val_roc_auc", fold_auc)
                mlflow.log_metric("val_brier_score", fold_brier)

        agg_auc = roc_auc_score(all_labels, all_preds)

        mlflow.log_metric("mean_val_roc_auc", np.mean(fold_aucs))
        mlflow.log_metric("mean_val_brier_score", np.mean(fold_briers))
        mlflow.log_metric("agg_val_roc_auc", agg_auc)
        mlflow.log_params(model.get_params())

        with plt.style.context("dark_background"):
            fig, ax = plt.subplots()
            fpr, tpr, _ = roc_curve(all_labels, all_preds)
            ax.plot(fpr, tpr, label=f"ROC curve (AUC = {agg_auc:.2f})")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC Curve - LightGBM (all folds)")
            ax.legend(loc="lower right")
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)

        output_sample = pd.DataFrame(
            model.predict_proba(X)[:, 1],
            columns=["podium_probability"]
        )
        signature = infer_signature(X, output_sample)
        mlflow.lightgbm.log_model(model, name=config["run_name"], signature=signature)

    return model, all_preds, all_labels