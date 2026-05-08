from pydantic import BaseModel

class RacePredictionRequest(BaseModel):
    year: int
    round: int

class DriverPrediction(BaseModel):
    driverId: int
    driver: str
    podium_probability: float