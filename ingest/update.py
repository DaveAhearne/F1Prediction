import requests
import pandas as pd
import io
import lakefs
import pandas as pd
from lakefs.client import Client
from f1_predictor.common.config import settings

class LakeFSRepository():
    def __init__(self):
        self._clt = Client(
            host=settings.lakefs_host,
            username=settings.lakefs_installation_access_key_id,
            password=settings.lakefs_installation_secret_access_key,
        )

        self._branch = lakefs.Repository(settings.lakefs_repo, client=self._clt).branch("main")

    def get_commit_sha(self) -> str:
        return self._branch.head.id

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

class JolpicaRepository():
    API_URL_BASE = "https://api.jolpi.ca/ergast/f1"

    def __init__(self, year, round):
        self.year = year
        self.round = round
    
    def get_api_data(self, year, section):
        response = requests.get(f"{self.API_URL_BASE}/{year}/{section}?format=json")
        response.raise_for_status()
        return response.json()
    
    def get_constructor_data(self) -> pd.DataFrame:
        data = self.get_api_data(self.year, "constructors")["MRData"]["ConstructorTable"]["Constructors"]
        return pd.json_normalize(data)

    def get_race_result_data(self):
        return self.get_api_data(self.year, f"{self.round}/results")["MRData"]["RaceTable"]["Races"][0]

def run():
    print(JolpicaRepository(2025,1).get_constructor_data())
