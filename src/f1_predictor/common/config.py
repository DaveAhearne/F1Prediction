from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    lakefs_host: str
    lakefs_username: str
    lakefs_password: str
    lakefs_repo: str = "f1-race-data"
    
    host: str = "0.0.0.0"
    port: int = 1234
    workers: int = 1

settings = Settings()