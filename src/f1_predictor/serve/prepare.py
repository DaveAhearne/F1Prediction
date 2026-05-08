import pandas as pd
import f1_predictor.features.context as context_features

def build_race_features(df: pd.DataFrame, races: pd.DataFrame, circuits: pd.DataFrame, year: int, round: int) -> pd.DataFrame:
    target_df = df[(df["year"] == year) & (df["round"] == round)]

    if not target_df.empty:
        return target_df
    
    last_known_result = df[df["raceId"] == df.sort_values(by=["year", "round"])["raceId"].iloc[-1]]
    target_race_result = races[(races["year"] == year) & (races["round"] == round)][["raceId", "circuitId", "date", "year", "round"]]

    extrapolated_last_known_result = last_known_result.copy().assign(
        raceId=target_race_result["raceId"].iloc[0],
        circuitId=target_race_result["circuitId"].iloc[0],
        date=target_race_result["date"].iloc[0],
        year=year,
        round=round
    )

    extrapolated_last_known_result = extrapolated_last_known_result.merge(
        circuits[["circuitId","country"]],
        on="circuitId"
    )

    extrapolated_last_known_result = context_features.add_home_race(extrapolated_last_known_result)
    extrapolated_last_known_result = context_features.add_circuit_type(extrapolated_last_known_result)
    extrapolated_last_known_result = context_features.add_regulation_era(extrapolated_last_known_result)

    return extrapolated_last_known_result