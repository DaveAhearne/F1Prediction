from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    lakefs_host: str
    lakefs_installation_access_key_id:str
    lakefs_installation_secret_access_key:str
    lakefs_repo: str = "f1-race-data"

    mlflow_tracking_uri: str
    mlflow_experiment_name: str

    train_n_estimators: int = 0
    train_learning_rate: float = 0.0
    train_num_leaves: int = 0
    train_min_child_samples: int = 0
    train_scale_pos_weight: float = 0.0
    train_subsample: float = 0.0
    train_colsample_bytree: float = 0.0
    
    host: str = "0.0.0.0"
    port: int = 1234
    workers: int = 1

    log_level: str = "INFO"

settings = Settings()