import pandas as pd

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Just tidying up some of the names from the merge
    data = data.rename(columns={"name_race": "race_name", "name_constructor": "constructor_name"})

    # This is the target variable, there's an argument for moving it into feature engineering
    data["podiumFinish"] = (data["position"] <= 3).astype(int)

    # The data pre-1990 has huge consistency problems, and doesn't provide useful signal so we drop it
    data = data[data["year"] > 1990]

    # 2025 is the test set
    data = data[data["year"] < 2025]

    # We engineer the target variable, past that point we don't need this anymore and it contains a ton of NaN's
    # that represent DNF's in various forms so we can remove it
    data = data.drop(columns=["position"])

    return data