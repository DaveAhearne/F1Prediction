from pydantic_settings import BaseSettings,SettingsConfigDict

class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    lakefs_host: str
    lakefs_username: str
    lakefs_password: str
    lakefs_repo: str = "f1-race-data"
    data_dir: str = "data/raw"

settings = IngestSettings()