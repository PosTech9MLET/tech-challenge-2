"""Configuração da aplicação carregadas do .env."""

from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient
from pydantic_settings import BaseSettings, SettingsConfigDict

METRICS: list[str] = ["accuracy", "precision", "recall", "f1"]


def get_secret_from_keyvault(vault_name: str, secret_name: str) -> str:
    """Busca um secret do Azure Key Vault.

    Args:
        vault_name: Nome do Key Vault.
        secret_name: Nome do secret.

    Returns:
        Valor do secret.
    """
    vault_url = f"https://{vault_name}.vault.azure.net"
    credential = AzureCliCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client.get_secret(secret_name).value


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

    # azure / dvc
    azure_storage_account: str = ""
    azure_container_name: str = ""
    azure_storage_key: str = ""
    azure_keyvault_name: str = "techchallengevaults"

    # mlflow
    mlflow_tracking_uri: str = ""
    mlflow_artifact_location: str = ""

    def get_azure_storage_key(self) -> str:
        """Retorna a access key do storage, buscando do Key Vault se necessário.

        Returns:
            Access key do Azure Storage.
        """
        if not self.azure_storage_key:
            return get_secret_from_keyvault(
                self.azure_keyvault_name, "AZURE-STORAGE-KEY"
            )
        return self.azure_storage_key


settings = Settings()
