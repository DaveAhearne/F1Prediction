import pandas as pd

def build_race_frame(
    races: pd.DataFrame,
    results: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
    statuses: pd.DataFrame,
    circuits: pd.DataFrame,
) -> pd.DataFrame:
    df = pd.merge(
        left=results[["resultId", "raceId", "driverId", "constructorId", "position", "statusId", "points"]],
        right=races[["raceId", "year", "round", "circuitId", "name", "date"]],
        on="raceId"
    )
    df = pd.merge(df, drivers[["driverId", "dob", "nationality"]], on="driverId")
    df = pd.merge(df, constructors[["constructorId", "name"]], on="constructorId", suffixes=("_race", "_constructor"))
    df = pd.merge(df, statuses[["statusId", "status"]], on="statusId")
    df = pd.merge(df, circuits[["circuitId", "location", "country"]], on="circuitId")

    return df