"""Préprocessamento de dados."""

import os
import warnings

import pandas as pd


def load_orders(file_path: str) -> pd.DataFrame:
    """Carrega os dados de pedidos.

    Args:
        file_path: Caminho para o arquivos orders.csv.

    Returns:
        DataFrame com os pedidos tratados.
    """
    orders = pd.read_csv(file_path)

    # cria uma flag is_first_order para indicar se é o primeiro pedido do cliente
    orders["is_first_order"] = orders["days_since_prior_order"].isna().astype(int)

    # preenche os valores faltantes de days_since_prior_order com a mediana
    median = orders["days_since_prior_order"].median()
    orders["days_since_prior_order"] = orders["days_since_prior_order"].fillna(median)

    return orders


def load_products(file_path: str) -> pd.DataFrame:
    """Carrega os dados de produtos.

    Args:
        file_path: Caminho para o arquivos products.csv.

    Returns:
        DataFrame com os produtos tratados.
    """
    products = pd.read_csv(file_path)

    # padroniza o nome dos produtos para facilitar a análise
    products["product_name"] = products["product_name"].str.strip().str.lower()

    return products


def load_aisles(file_path: str) -> pd.DataFrame:
    """Carrega os dados dos corredores.

    Args:
        file_path: Caminho para o arquivos aisles.csv.

    Returns:
        DataFrame com os corredores tratados.
    """
    aisles = pd.read_csv(file_path)

    # valida se há nulos
    if aisles["aisle"].isnull().any():
        warnings.warn("Atenção: Existem corredores com nome nulo. Verifique os dados.")

    return aisles


def load_departments(file_path: str) -> pd.DataFrame:
    """Carrega os dados dos departamentos.

    Args:
        file_path: Caminho para o arquivos departments.csv.

    Returns:
        DataFrame com os departamentos tratados.
    """
    departments = pd.read_csv(file_path)

    # valida se há nulos
    if departments["department"].isnull().any():
        warnings.warn(
            "Atenção: Existem departamentos com nome nulo. Verifique os dados."
        )

    return departments


def load_order_products(data_dir: str, subset: str) -> pd.DataFrame:
    """Carrega os dados de produtos por pedido.

    Args:
        data_dir: Diretório onde os arquivos estão localizados.
        subset: Subconjunto dos dados (prior, train ou test).

    Returns:
        DataFrame com os produtos por pedido tratados.
    """
    # define valores padrao
    subsets_mapping = ["prior", "train"]

    if subset not in subsets_mapping:
        raise ValueError(f"Subset inválido: '{subset}'. Escolha 'prior' ou 'train'.")

    # une o caminho e busca automaticamente o arquivo correto com base no subset
    data_dir = os.path.join(data_dir, f"order_products__{subset}.csv")
    order_products = pd.read_csv(data_dir)

    return order_products
