from typing import Optional

import pandas as pd

def clean_data(data: pd.DataFrame, min_year=1990, max_year: Optional[int] = None) -> pd.DataFrame:
    # Just tidying up some of the names from the merge
    data = data.rename(columns={"name_race": "race_name", "name_constructor": "constructor_name"})

    # This is the target variable, there's an argument for moving it into feature engineering
    data["podiumFinish"] = (data["position"] <= 3).astype(int)

    data = data[data["year"] > min_year]
    if max_year is not None:
        data = data[data["year"] < max_year]

    # We engineer the target variable, past that point we don't need this anymore and it contains a ton of NaN's
    # that represent DNF's in various forms so we can remove it
    data = data.drop(columns=["position"])

    return data