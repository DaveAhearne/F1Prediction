import pandas as pd
import pandera.pandas as pa
from pandera.typing import DateTime, Category

class F1ValidationSchema(pa.DataFrameModel):
    year: int = pa.Field(gt=1990)
    raceId: int = pa.Field(gt=0)
    resultId: int = pa.Field(gt=0)
    driverId: Category
    constructorId: Category
    circuitId: Category
    statusId: int = pa.Field(gt=0)
    round: int = pa.Field(gt=0,le=30)
    date: DateTime = pa.Field(ge=pd.Timestamp('1990-01-01'))
    dob: DateTime = pa.Field(gt=pd.Timestamp('1940-01-01'), lt=pd.Timestamp('2010-01-01'))
    nationality: str
    race_name: str
    location: str
    country: str
    status: str
    constructor_name: str
    podiumFinish: int = pa.Field(isin=[1,0])

    # Only one driver should be able to race in a single race at once
    class Config:
        unique = ["raceId", "driverId"]

def check_schema(data: pd.DataFrame) -> None:
    try:
        F1ValidationSchema.validate(data)
        print("Schema validation passed")
    except pa.errors.SchemaError as e:
        print(f"Schema validation FAILED:\n{e.failure_cases}")
        raise