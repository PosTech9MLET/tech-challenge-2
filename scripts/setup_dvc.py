"""Configura o DVC remote local com a account key do Azure Key Vault."""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import settings


def setup_dvc_remote() -> None:
    """Busca a account key do Key Vault e configura o DVC remote local.

    Raises:
        subprocess.CalledProcessError: Se o comando DVC falhar.
    """
    account_key = settings.get_azure_storage_key()

    subprocess.run(
        [
            "dvc",
            "remote",
            "modify",
            "--local",
            "azure_remote",
            "account_key",
            account_key,
        ],
        check=True,
    )
    print("✓ DVC remote configurado com sucesso.")


def main() -> None:
    """Ponto de entrada do script."""
    setup_dvc_remote()


if __name__ == "__main__":
    main()
