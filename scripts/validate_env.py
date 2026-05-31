"""Script para validação do ambiente de desenvolvimento."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import settings


def check_imports() -> bool:
    """Verifica se as dependências estão instaladas corretamente.

    Returns:
        True se todos os imports funcionarem, False caso contrário.
    """
    packages = [
        "torch",
        "sklearn",
        "mlflow",
        "dvc",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
    ]
    success = True
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - Não encontrado!")
            success = False
    return success


def check_settings_variables() -> bool:
    """Verifica se as variáveis do Settings estão definidas corretamente.

    Returns:
        True se todas as variáveis estiverem definidas, False caso contrário.
    """
    required_vars = [
        "seed",
        "test_size",
        "train_size",
        "data_input_path",
        "data_output_path",
        "early_stopping_patience",
        "azure_storage_account",
        "azure_storage_container",
        "azure_storage_key",
        "mlflow_tracking_uri",
        "mlflow_artifact_location",
    ]

    success = True
    for var in required_vars:
        if getattr(settings, var) in [None, "", 0]:
            print(f"✗ {var} - Variável não definida ou vazia!")
            success = False
        else:
            print(f"✓ {var}")
    return success


def check_paths() -> bool:
    """Verifica se os caminhos de entrada e saída estão acessíveis.

    Returns:
        True se ambos os caminhos forem acessíveis, False caso contrário.
    """
    success = True
    for path in [settings.data_input_path, settings.data_output_path]:
        if not os.path.exists(path):
            print(f"✗ {path} - Caminho não encontrado!")
            success = False
        else:
            print(f"✓ {path}")
    return success


def main() -> None:
    """Executa as validações do ambiente."""
    print("Validando ambiente de desenvolvimento...\n")

    imports_ok = check_imports()
    settings_ok = check_settings_variables()
    paths_ok = check_paths()

    if imports_ok and settings_ok and paths_ok:
        print("\n✓ Ambiente de desenvolvimento validado com sucesso!")
    else:
        print("\n✗ Erros encontrados. Verifique as mensagens acima.")
        sys.exit(1)


if __name__ == "__main__":
    main()
