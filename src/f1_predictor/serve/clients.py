import io
import lakefs
import mlflow
import pandas as pd
from lakefs.client import Client
from f1_predictor.common.config import settings

class MLFlowClient():
    def get_model(self, experiment_name, tag):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        client = mlflow.MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        model_version = client.get_model_version_by_alias(experiment_name, tag)
        model = mlflow.onnx.load_model(f"models:/{experiment_name}@{tag}")
        return model, model_version.version, model_version.run_id

class LakeFSClient():
    def __init__(self):
        self._clt = Client(
            host=settings.lakefs_host,
            username=settings.lakefs_installation_access_key_id,
            password=settings.lakefs_installation_secret_access_key,
        )
        self._branch = lakefs.Repository(settings.lakefs_repo, client=self._clt).branch("main")

    def _read_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        with self._branch.object(filepath).reader(mode="rb") as f:
            return pd.read_csv(io.BytesIO(f.read()), **kwargs)

    def load_races(self) -> pd.DataFrame:
        return self._read_csv("raw/races.csv", na_values="\\N", parse_dates=["date", "quali_date"])

    def load_results(self) -> pd.DataFrame:
        return self._read_csv("raw/results.csv", na_values="\\N")

    def load_drivers(self) -> pd.DataFrame:
        return self._read_csv("raw/drivers.csv", na_values="\\N", parse_dates=["dob"])

    def load_constructors(self) -> pd.DataFrame:
        return self._read_csv("raw/constructors.csv")

    def load_statuses(self) -> pd.DataFrame:
        return self._read_csv("raw/status.csv")

    def load_circuits(self) -> pd.DataFrame:
        return self._read_csv("raw/circuits.csv")