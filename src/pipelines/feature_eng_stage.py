"""Stage 2 do pipeline DVC — feature engineering e geração dos splits."""

import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from configs.settings import settings
from src.features.feature_builder import (
    build_product_features,
    build_user_features,
    build_user_product_features,
    split_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os 3 Parquets gerados pelo stage de pré-processamento.

    Returns:
        Tupla (orders, products_full, interactions).
    """
    log.info("Carregando dados processados de %s...", PROCESSED_DIR)
    orders = pd.read_parquet(PROCESSED_DIR / "orders_clean.parquet")
    products_full = pd.read_parquet(PROCESSED_DIR / "products_full.parquet")
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
    return orders, products_full, interactions


def build_all_features(
    orders: pd.DataFrame,
    products_full: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Constrói e persiste as 3 tabelas de features.

    Args:
        orders: Pedidos limpos.
        products_full: Produtos enriquecidos.
        interactions: Tabela unificada de interações.

    Returns:
        Tupla (user_features, product_features, user_product_features).
    """
    log.info("Construindo features de usuário...")
    user_features = build_user_features(orders=orders, interactions=interactions)
    user_features.to_parquet(FEATURES_DIR / "user_features.parquet", index=False)

    log.info("Construindo features de produto...")
    product_features = build_product_features(
        products_full=products_full, interactions=interactions
    )
    product_features.to_parquet(FEATURES_DIR / "product_features.parquet", index=False)

    log.info("Construindo features de par user-produto...")
    user_product_features = build_user_product_features(
        interactions=interactions, orders=orders
    )
    user_product_features.to_parquet(
        FEATURES_DIR / "user_product_features.parquet", index=False
    )
    return user_features, product_features, user_product_features


def save_splits_and_encoders(
    user_features: pd.DataFrame,
    product_features: pd.DataFrame,
    user_product_features: pd.DataFrame,
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gera os splits train/val/test e salva os encoders de ID.

    Args:
        user_features: Features de usuário.
        product_features: Features de produto.
        user_product_features: Features do par user-produto.
        interactions: Tabela de interações completa.

    Returns:
        Tupla (train, val, test).
    """
    log.info("Criando splits train/val/test (seed=%d)...", settings.seed)
    train, val, test, user_enc, prod_enc = split_dataset(
        user_features=user_features,
        product_features=product_features,
        user_product_features=user_product_features,
        interactions=interactions,
        seed=settings.seed,
    )
    train.to_parquet(FEATURES_DIR / "train.parquet", index=False)
    val.to_parquet(FEATURES_DIR / "val.parquet", index=False)
    test.to_parquet(FEATURES_DIR / "test.parquet", index=False)

    log.info("Salvando encoders de user_id e product_id...")
    encoders = {"user_encoder": user_enc, "product_encoder": prod_enc}
    with open(MODELS_DIR / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    return train, val, test


def save_metrics(
    user_features: pd.DataFrame,
    product_features: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Calcula e salva as métricas do stage em um JSON.

    Args:
        user_features: Features de usuário.
        product_features: Features de produto.
        train: Split de treino.
        val: Split de validação.
        test: Split de teste.

    Returns:
        Dicionário com as métricas calculadas.
    """
    metrics = {
        "n_user_features": int(user_features.shape[1]),
        "n_product_features": int(product_features.shape[1]),
        "n_train_samples": int(len(train)),
        "n_val_samples": int(len(val)),
        "n_test_samples": int(len(test)),
        "reorder_rate_train": float(train["reordered"].mean()),
    }
    with open(FEATURES_DIR / "feature_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def run() -> None:
    """Constrói features e cria splits train/val/test."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    orders, products_full, interactions = load_processed_data()
    user_features, product_features, user_product_features = build_all_features(
        orders, products_full, interactions
    )
    train, val, test = save_splits_and_encoders(
        user_features, product_features, user_product_features, interactions
    )
    metrics = save_metrics(user_features, product_features, train, val, test)

    log.info("Stage 2 concluído. %s", metrics)


if __name__ == "__main__":
    run()
