import pandas as pd
from f1_predictor.data.clean import clean_data
from f1_predictor.data.validate import validate_data
from f1_predictor.data.load import load_races, load_circuits, load_constructors, load_drivers, load_results, load_statuses
from f1_predictor.data.merge import build_race_frame

def run_pipeline() -> pd.DataFrame:
    raw = build_race_frame(
        races=load_races(),
        circuits=load_circuits(),
        constructors=load_constructors(),
        drivers=load_drivers(),
        results=load_results(),
        statuses=load_statuses()
    )

    cleaned = clean_data(raw)
    validated = validate_data(cleaned)

    return validated