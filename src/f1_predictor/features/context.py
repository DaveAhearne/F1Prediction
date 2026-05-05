import pandas as pd

def add_season_podium_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "round"]).copy()
    df["driver_season_podium_rate"] = (
        df.groupby(["year", "driverId"], observed=True)["podiumFinish"]
        .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(0)
    return df

def add_championship_position(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["year", "round"]).copy()

    df["cumulative_points"] = (
        df.groupby(["year", "driverId"], observed=True)["points"]
        .transform(lambda x: x.shift(1).cumsum())
    )

    df["driver_championship_position"] = (
        df.groupby("raceId")["cumulative_points"]
        .rank(ascending=False, method="min")
    )

    df = df.drop(columns=["cumulative_points", "points"])

    df["driver_championship_position"] = (
        df.groupby("raceId")["driver_championship_position"]
        .transform(lambda x: x.fillna(len(x)))
    )

    return df

def add_circuit_type(df: pd.DataFrame) -> pd.DataFrame:
    circuit_type_map = {
        6: "street", 12: "street", 15: "street", 29: "street", 33: "street",
        37: "street", 42: "street", 43: "street", 44: "street", 59: "street",
        67: "street", 73: "street", 77: "street", 80: "street", 81: "street",
        71: "hybrid", 79: "hybrid",
        1: "permanent", 2: "permanent", 3: "permanent", 4: "permanent",
        5: "permanent", 7: "permanent", 8: "permanent", 9: "permanent",
        10: "permanent", 11: "permanent", 13: "permanent", 14: "permanent",
        16: "permanent", 17: "permanent", 18: "permanent", 19: "permanent",
        20: "permanent", 21: "permanent", 22: "permanent", 24: "permanent",
        25: "permanent", 26: "permanent", 27: "permanent", 28: "permanent",
        30: "permanent", 31: "permanent", 32: "permanent", 34: "permanent",
    }
    df["circuit_type"] = df["circuitId"].map(circuit_type_map)
    return df

def add_home_race(df: pd.DataFrame) -> pd.DataFrame:
    nationality_country_map = {
        "British": "UK", "German": "Germany", "Spanish": "Spain",
        "Finnish": "Finland", "Brazilian": "Brazil", "Australian": "Australia",
        "French": "France", "Italian": "Italy", "Austrian": "Austria",
        "Dutch": "Netherlands", "Belgian": "Belgium", "Canadian": "Canada",
        "American": "USA", "Japanese": "Japan", "New Zealander": "New Zealand",
        "Mexican": "Mexico", "Monegasque": "Monaco", "Swiss": "Switzerland",
        "Danish": "Denmark", "Swedish": "Sweden", "Argentine": "Argentina",
        "Portuguese": "Portugal", "South African": "South Africa",
        "Colombian": "Colombia", "Venezuelan": "Venezuela", "Czech": "Czech Republic",
        "Hungarian": "Hungary", "Russian": "Russia", "Polish": "Poland",
        "Malaysian": "Malaysia", "Indian": "India", "Chinese": "China",
        "Thai": "Thailand",
    }
    df["driver_country"] = df["nationality"].map(nationality_country_map)
    df["is_home_race"] = (df["driver_country"] == df["country"]).astype(int)
    df = df.drop(columns=["driver_country", "country"])
    return df


def add_regulation_era(df: pd.DataFrame) -> pd.DataFrame:
    regulation_era_map = {year: era for era, years in {
        "v10": range(1991, 2006),
        "v8": range(2006, 2014),
        "hybrid": range(2014, 2022),
        "ground_effect": range(2022, 2026),
        "battery": range(2026, 2030),
    }.items() for year in years}

    df["regulation_era"] = df["year"].map(regulation_era_map)
    return df

def add_grid_size(df: pd.DataFrame):
    df["grid_size"] = df.groupby("raceId")["driverId"].transform("nunique")
    return df
