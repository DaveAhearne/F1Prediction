from pydantic_settings import BaseSettings,SettingsConfigDict

class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    lakefs_host: str
    lakefs_installation_access_key_id:str
    lakefs_installation_secret_access_key:str
    lakefs_repo: str = "f1-race-data"

settings = IngestSettings()