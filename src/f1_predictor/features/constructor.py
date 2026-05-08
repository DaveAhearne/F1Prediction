import pandas as pd

def add_constructor_dnf_rates(df: pd.DataFrame) -> pd.DataFrame:
    non_mechanical_statuses = [
        "Finished",
        "+1 Lap", "+2 Laps", "+3 Laps", "+4 Laps", "+5 Laps",
        "+6 Laps", "+7 Laps", "+8 Laps", "+9 Laps", "+10 Laps",
        "+11 Laps", "+12 Laps", "+13 Laps", "+14 Laps", "+15 Laps",
        "+16 Laps", "+17 Laps", "+18 Laps", "+19 Laps", "+20 Laps",
        "+21 Laps", "+22 Laps", "+23 Laps", "+24 Laps", "+25 Laps",
        "+26 Laps", "+29 Laps", "+30 Laps", "+38 Laps", "+42 Laps",
        "+44 Laps", "+46 Laps", "+49 Laps",
        "Accident", "Collision", "Collision damage", "Spun off", "Fatal accident",
        "Physical", "Injured", "Injury", "Driver unwell", "Eye injury",
        "Illness", "Safety belt", "Driver Seat", "Seat",
        "Disqualified", "Excluded", "Underweight", "107% Rule",
        "Did not qualify", "Did not prequalify", "Not classified",
    ]
    mechanical_ids = set(
        df[~df["status"].isin(non_mechanical_statuses)]["statusId"]
    )

    df = df.sort_values(["constructorId", "year", "round"]).copy()
    df["mechanicalDNF"] = df["statusId"].isin(mechanical_ids).astype(int)
    df["constructor_mechanical_dnf_rate_5"] = (
        df.groupby("constructorId", observed=True)["mechanicalDNF"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    ).fillna(0)
    df = df.drop(columns=["mechanicalDNF"])
    return df

def add_constructor_rolling_podium_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["constructorId", "year", "round"]).copy()
    for window in [3, 5, 10]:
        col = f"constructor_podium_rate_{window}"
        df[col] = (
            df.groupby("constructorId", observed=True)["podiumFinish"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        ).fillna(0)
    return df
