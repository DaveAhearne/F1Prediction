import requests
import pandas as pd
import io
import lakefs
import pandas as pd
from lakefs.client import Client
from f1_predictor.common.config import settings
from dataclasses import dataclass

@dataclass()
class F1Data:
    results: pd.DataFrame
    constructors: pd.DataFrame
    races: pd.DataFrame
    statuses: pd.DataFrame
    drivers: pd.DataFrame
    circuits: pd.DataFrame

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

    def get_api_data(self, arguments):
        response = requests.get(f"{self.API_URL_BASE}/{arguments}?format=json")
        response.raise_for_status()
        return response.json()
    
    def get_driver_data(self, year) -> pd.DataFrame:
        data = self.get_api_data(f"{year}/drivers")["MRData"]["DriverTable"]["Drivers"]
        return pd.json_normalize(data)
    
    def get_constructor_data(self, year) -> pd.DataFrame:
        data = self.get_api_data(f"{year}/constructors")["MRData"]["ConstructorTable"]["Constructors"]
        return pd.json_normalize(data)

    def get_race_data(self, year) -> pd.DataFrame:
        data = self.get_api_data(f"{year}/races")["MRData"]["RaceTable"]["Races"]
        return pd.json_normalize(data)
    
    def get_status_data(self) -> pd.DataFrame:
        data = self.get_api_data("status")["MRData"]["StatusTable"]["Status"]
        return pd.json_normalize(data)

    def get_circuit_data(self) -> pd.DataFrame:
        data = self.get_api_data("circuits")["MRData"]["CircuitTable"]["Circuits"]
        return pd.json_normalize(data)

    def get_next_race_results(self, year, round):
        data = self.get_api_data(f"{year}/{round}/results")
        
        if data["MRData"]["RaceTable"]["Races"]:
            return year, round, pd.json_normalize(data["MRData"]["RaceTable"]["Races"])
        
        data = self.get_api_data(year + 1, "1/results")
        
        if data["MRData"]["RaceTable"]["Races"]:
            return year + 1, 1, pd.json_normalize(data["MRData"]["RaceTable"]["Races"])
        
        return None, None, None

def get_data_from_lake()-> F1Data:
    lakefs_repo = LakeFSRepository()

    drivers = lakefs_repo.load_drivers()
    constructors = lakefs_repo.load_constructors()
    circuits = lakefs_repo.load_circuits()
    status = lakefs_repo.load_statuses()
    race = lakefs_repo.load_races()
    res = lakefs_repo.load_results()

    return F1Data(res, constructors, race, status, drivers, circuits)

def get_data_from_api(lakefsData: F1Data) -> F1Data:
    f1_api_repo = JolpicaRepository()

    last_row = pd.merge(
        left=lakefsData.results,
        right=lakefsData.races,
        how="left",
        on="raceId"
    ).sort_values(["year", "round"]).iloc[-1]

    latest_year  = last_row["year"]
    latest_round = last_row["round"] + 1

    latest_year, latest_round, race_result_data = f1_api_repo.get_next_race_results(latest_year, latest_round)
    constructor_data = f1_api_repo.get_constructor_data(latest_year)    
    status_data = f1_api_repo.get_status_data()    
    race_data = f1_api_repo.get_race_data(latest_year)    
    driver_data = f1_api_repo.get_driver_data(latest_year)    
    circuit_data = f1_api_repo.get_circuit_data()    

    return F1Data(race_result_data, constructor_data, race_data, status_data, driver_data, circuit_data)

def run():
    lake_data = get_data_from_lake()
    api_data = get_data_from_api(lake_data)

    print(api_data)
    
