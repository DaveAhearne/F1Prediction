from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    lakefs_host: str
    lakefs_username: str
    lakefs_password: str
    lakefs_repo: str = "f1-race-data"

    train_n_estimators: int
    train_learning_rate: float
    train_num_leaves: int
    train_min_child_samples: int
    train_scale_pos_weight: float
    train_subsample: float
    train_colsample_bytree: float
    
    host: str = "0.0.0.0"
    port: int = 1234
    workers: int = 1

settings = Settings()