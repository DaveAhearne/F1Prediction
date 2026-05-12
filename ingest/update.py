import requests
import pandas as pd
import io
import lakefs
import pandas as pd
from lakefs.client import Client
from f1_predictor.common.config import settings
from dataclasses import dataclass

from f1_predictor.data import clean, validate
from f1_predictor.data.merge import build_race_frame

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

        self._repo = lakefs.Repository(settings.lakefs_repo, client=self._clt)
        self._branch = self._repo.branch("main")

    def update_branch(self, name: str):
        self._branch = self._repo.branch(name)

    def create_branch(self, name: str, source: str = "main"):
        branch = self._repo.branch(name).create(source_reference=source, exist_ok=True)
        return branch

    def merge_into(self, source: str, target: str = "main", message: str = "Merge branch"):
        self._repo.branch(source).merge_into(target, message=message)

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
            return year, round, pd.json_normalize(
                data["MRData"]["RaceTable"]["Races"],
                record_path=["Results"],
                meta=["season", "round", "raceName", "date"]
            )
        
        data = self.get_api_data(f"{year + 1}/1/results")
        
        if data["MRData"]["RaceTable"]["Races"]:
            return year + 1, 1, pd.json_normalize(
                data["MRData"]["RaceTable"]["Races"],
                record_path=["Results"],
                meta=["season", "round", "raceName", "date"]
            )
        
        return None, None, None

def get_data_from_lake(lakefs_repo: LakeFSRepository)-> F1Data:
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

    if latest_year is None:
        return None, None, None

    constructor_data = f1_api_repo.get_constructor_data(latest_year)    
    status_data = f1_api_repo.get_status_data()    
    race_data = f1_api_repo.get_race_data(latest_year)    
    driver_data = f1_api_repo.get_driver_data(latest_year)    
    circuit_data = f1_api_repo.get_circuit_data()    

    return latest_year, latest_round, F1Data(race_result_data, constructor_data, race_data, status_data, driver_data, circuit_data)

def reconcile(lake_data: F1Data, api_data: F1Data, latest_round: int) -> F1Data:
    new_drivers = api_data.drivers[
        ~api_data.drivers["driverId"].isin(lake_data.drivers["driverRef"])
    ].rename(columns={
        "driverId":    "driverRef",
        "givenName":   "forename",
        "familyName":  "surname",
        "dateOfBirth": "dob",
    })

    new_drivers["driverId"] = range(
        lake_data.drivers["driverId"].max() + 1,
        lake_data.drivers["driverId"].max() + 1 + len(new_drivers)
    )

    new_drivers = new_drivers[[c for c in lake_data.drivers.columns if c in new_drivers.columns]]

    drivers = pd.concat([lake_data.drivers, new_drivers], ignore_index=True)

    new_constructors = api_data.constructors[
        ~api_data.constructors["constructorId"].isin(lake_data.constructors["constructorRef"])
    ].rename(columns={"constructorId": "constructorRef"})

    new_constructors["constructorId"] = range(
        lake_data.constructors["constructorId"].max() + 1,
        lake_data.constructors["constructorId"].max() + 1 + len(new_constructors)
    )
    constructors = pd.concat([lake_data.constructors, new_constructors], ignore_index=True)

    new_statuses = api_data.statuses[
        ~api_data.statuses["status"].isin(lake_data.statuses["status"])
    ].copy()

    new_statuses["statusId"] = range(
        lake_data.statuses["statusId"].max() + 1,
        lake_data.statuses["statusId"].max() + 1 + len(new_statuses)
    )

    statuses = pd.concat([lake_data.statuses, new_statuses], ignore_index=True)

    driver_lookup      = dict(zip(drivers["driverRef"],          drivers["driverId"]))
    constructor_lookup = dict(zip(constructors["constructorRef"], constructors["constructorId"]))
    circuit_lookup     = dict(zip(lake_data.circuits["circuitRef"], lake_data.circuits["circuitId"]))
    status_lookup      = dict(zip(statuses["status"],             statuses["statusId"]))

    new_race = api_data.races[api_data.races["round"].astype(int) == latest_round].copy()

    new_race["round"] = new_race["round"].astype(int)
    new_race["season"] = new_race["season"].astype(int)
    new_race["date"] = pd.to_datetime(new_race["date"])

    new_race = new_race.rename(columns={"season": "year"})

    new_race["raceId"]    = lake_data.races["raceId"].max() + 1
    new_race["circuitId"] = new_race["Circuit.circuitId"].map(circuit_lookup)

    new_race = new_race.rename(columns={"season": "year", "raceName": "name"})
    races = pd.concat([lake_data.races, new_race], ignore_index=True)

    results_flat = api_data.results.copy()

    results_flat["round"] = results_flat["round"].astype(int)
    results_flat["season"] = results_flat["season"].astype(int)
    results_flat["date"] = pd.to_datetime(results_flat["date"])
    
    race_id = new_race["raceId"].iloc[0]

    results_flat["raceId"]        = race_id
    results_flat["driverId"]      = results_flat["Driver.driverId"].map(driver_lookup)
    results_flat["constructorId"] = results_flat["Constructor.constructorId"].map(constructor_lookup)
    results_flat["statusId"]      = results_flat["status"].map(status_lookup)
    results_flat["position"]      = pd.to_numeric(results_flat["position"], errors="coerce")
    results_flat["points"]        = results_flat["points"].astype(float)
    results_flat["resultId"]      = range(
        lake_data.results["resultId"].max() + 1,
        lake_data.results["resultId"].max() + 1 + len(results_flat)
    )

    results = pd.concat([lake_data.results, results_flat], ignore_index=True)

    return F1Data(results, constructors, races, statuses, drivers, lake_data.circuits)

def write_to_lakefs(data: F1Data, branch):
    try:
        with branch.transact(commit_message=f"Update: add new race data") as tx:
            tx.object("raw/results.csv").upload(data.results.to_csv(index=False).encode())
            tx.object("raw/races.csv").upload(data.races.to_csv(index=False).encode())
            tx.object("raw/drivers.csv").upload(data.drivers.to_csv(index=False).encode())
            tx.object("raw/constructors.csv").upload(data.constructors.to_csv(index=False).encode())
            tx.object("raw/status.csv").upload(data.statuses.to_csv(index=False).encode())
        print("Transaction committed to main")
    except Exception as e:
        print(f"Transaction failed and rolled back: {e}")

def run():
    lakefsRepo = LakeFSRepository()
    
    lake_data = get_data_from_lake(lakefsRepo)
    latest_year, latest_round, api_data = get_data_from_api(lake_data)

    if latest_year is None:
        print("No new race data available")
        return

    print(f"Fetched data from: year {latest_year} round: {latest_round}")
    
    updated_data = reconcile(lake_data, api_data, latest_round)

    staging_branch = lakefsRepo.create_branch("staging")
    write_to_lakefs(updated_data, staging_branch)
    print("Written to staging branch")

    try:
        merged_data = build_race_frame(
            races=updated_data.races,
            circuits=updated_data.circuits,
            constructors=updated_data.constructors,
            drivers=updated_data.drivers,
            results=updated_data.results,
            statuses=updated_data.statuses
        )
        validate.check_schema(clean.clean_data(merged_data, min_year=1990))
    except Exception as e:
        print(f"Validation failed — data remains on staging branch for inspection: {e}")
        return

    lakefsRepo.merge_into(source="staging", target="main", message=f"Merge validated race data to main: {latest_year} round {latest_round}")
    print("Merged to main")