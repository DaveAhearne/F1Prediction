import pandas as pd

def add_driver_rolling_podium_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["driverId", "year", "round"]).copy()
    for window in [3, 5, 10]:
        col = f"driver_podium_rate_{window}"
        df[col] = (
            df.groupby("driverId", observed=True)["podiumFinish"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        ).fillna(0)
    return df

def add_driver_circuit_podium_rate(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_circuit_podium_rate"] = (
        df.groupby(["driverId", "circuitId"], observed=True)["podiumFinish"]
        .transform(lambda x: x.shift(1).expanding().mean())
    ).fillna(0)
    return df

def add_driver_experience(df: pd.DataFrame) -> pd.DataFrame:
    df["driver_experience"] = df.groupby("driverId", observed=True).cumcount().astype("float64")
    return df

def add_driver_age(df: pd.DataFrame) -> pd.DataFrame:
    df["driverAge"] = (df["date"] - df["dob"]).dt.days / 365.25
    return df
