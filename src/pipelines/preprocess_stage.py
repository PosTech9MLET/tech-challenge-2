"""Stage 1 do pipeline DVC — pré-processamento dos dados brutos."""

import json
import logging
from pathlib import Path

import pandas as pd

from configs.settings import settings
from src.features.preprocess import (
    load_aisles,
    load_departments,
    load_order_products,
    load_orders,
    load_products,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = (
    Path(settings.data_input_path) if settings.data_input_path else Path("data/raw")
)
PROCESSED_DIR = (
    Path(settings.data_output_path)
    if settings.data_output_path
    else Path("data/processed")
)


def build_products_full(
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame,
) -> pd.DataFrame:
    """Enriquece produtos com corredor, departamento e flag orgânico.

    Args:
        products: DataFrame de produtos (já limpo pelo load_products).
        aisles: DataFrame de corredores.
        departments: DataFrame de departamentos.

    Returns:
        DataFrame com colunas aisle, department e is_organic.
    """
    df = products.merge(aisles, on="aisle_id", how="left")
    df = df.merge(departments, on="department_id", how="left")
    df["is_organic"] = df["product_name"].str.contains("organic", case=False, na=False)
    return df


def build_interactions(
    order_products_prior: pd.DataFrame,
    order_products_train: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Une prior e train em uma tabela única de interações com contexto do pedido.

    Args:
        order_products_prior: Produtos dos pedidos históricos.
        order_products_train: Produtos dos pedidos de treino.
        orders: Pedidos tratados (saída do load_orders).

    Returns:
        DataFrame com todas as interações e coluna split (prior/train).
    """
    prior = order_products_prior.copy()
    prior["split"] = "prior"

    train = order_products_train.copy()
    train["split"] = "train"

    interactions = pd.concat([prior, train], ignore_index=True)
    interactions = interactions.merge(
        orders[
            [
                "order_id",
                "user_id",
                "order_number",
                "order_dow",
                "order_hour_of_day",
                "days_since_prior_order",
                "is_first_order",
            ]
        ],
        on="order_id",
        how="left",
    )
    return interactions


def run() -> None:
    """Executa o stage de pré-processamento e salva os artefatos."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Carregando CSVs brutos de %s...", RAW_DIR)
    orders = load_orders(str(RAW_DIR / "orders.csv"))
    products = load_products(str(RAW_DIR / "products.csv"))
    aisles = load_aisles(str(RAW_DIR / "aisles.csv"))
    departments = load_departments(str(RAW_DIR / "departments.csv"))
    order_products_prior = load_order_products(str(RAW_DIR), subset="prior")
    order_products_train = load_order_products(str(RAW_DIR), subset="train")

    log.info("Enriquecendo tabela de produtos...")
    products_full = build_products_full(products, aisles, departments)

    log.info("Construindo tabela de interações unificada...")
    interactions = build_interactions(
        order_products_prior, order_products_train, orders
    )

    log.info("Salvando Parquets em %s...", PROCESSED_DIR)
    orders.to_parquet(PROCESSED_DIR / "orders_clean.parquet", index=False)
    products_full.to_parquet(PROCESSED_DIR / "products_full.parquet", index=False)
    interactions.to_parquet(PROCESSED_DIR / "interactions.parquet", index=False)

    metrics = {
        "n_users": int(orders["user_id"].nunique()),
        "n_orders": int(orders["order_id"].nunique()),
        "n_products": int(products_full["product_id"].nunique()),
        "n_interactions_prior": int((interactions["split"] == "prior").sum()),
        "n_interactions_train": int((interactions["split"] == "train").sum()),
        "pct_first_orders": float(orders["is_first_order"].mean()),
    }
    with open(PROCESSED_DIR / "preprocess_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Stage 1 concluído. %s", metrics)


if __name__ == "__main__":
    run()
