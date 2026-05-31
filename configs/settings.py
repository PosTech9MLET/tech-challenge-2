"""Configuração da aplicação carregadas do .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict

METRICS: list[str] = ["accuracy", "precision", "recall", "f1"]


class Settings(BaseSettings):
    """Configurações da aplicação carregadas do .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # modelo
    seed: int = 42
    test_size: float = 0.3
    train_size: float = 0.7

    data_input_path: str = ""
    data_output_path: str = ""
    early_stopping_patience: int = 5

    # dvc
    azure_storage_account: str = ""
    azure_storage_container: str = ""
    azure_storage_key: str = ""

    # mlflow
    mlflow_tracking_uri: str = ""
    mlflow_artifact_location: str = ""


settings = Settings()
