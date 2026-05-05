import io
import lakefs
import pandas as pd
from lakefs.client import Client
from ingest.settings import settings

_clt = Client(
    host=settings.lakefs_host,
    username=settings.lakefs_username,
    password=settings.lakefs_password,
)

_branch = lakefs.Repository(settings.lakefs_repo, client=_clt).branch("main")

def _read_csv(filepath: str, **kwargs) -> pd.DataFrame:
    with _branch.object(filepath).reader(mode="rb") as f:
        return pd.read_csv(io.BytesIO(f.read()), **kwargs)

def load_races() -> pd.DataFrame:
    return _read_csv("raw/races.csv", na_values="\\N", parse_dates=["date", "quali_date"])

def load_results() -> pd.DataFrame:
    return _read_csv("raw/results.csv", na_values="\\N")

def load_drivers() -> pd.DataFrame:
    return _read_csv("raw/drivers.csv", na_values="\\N", parse_dates=["dob"])

def load_constructors() -> pd.DataFrame:
    return _read_csv("raw/constructors.csv")

def load_statuses() -> pd.DataFrame:
    return _read_csv("raw/status.csv")

def load_circuits() -> pd.DataFrame:
    return _read_csv("raw/circuits.csv")