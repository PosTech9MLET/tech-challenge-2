"""Stage 2 do pipeline DVC — feature engineering e geração dos splits."""

import json
import logging
from pathlib import Path

import pandas as pd
from src.features.feature_builder import (
    build_product_features,
    build_user_features,
    build_user_product_features,
    split_dataset,
)

from configs.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
FEATURES_DIR = Path("data/features")


def run() -> None:
    """Constrói features e cria splits train/val/test."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Carregando dados processados de %s...", PROCESSED_DIR)
    orders = pd.read_parquet(PROCESSED_DIR / "orders_clean.parquet")
    products_full = pd.read_parquet(PROCESSED_DIR / "products_full.parquet")
    interactions = pd.read_parquet(PROCESSED_DIR / "interactions.parquet")

    log.info("Construindo features de usuário...")
    user_features = build_user_features(orders=orders, interactions=interactions)
    user_features.to_parquet(FEATURES_DIR / "user_features.parquet", index=False)

    log.info("Construindo features de produto...")
    product_features = build_product_features(
        products_full=products_full,
        interactions=interactions,
    )
    product_features.to_parquet(FEATURES_DIR / "product_features.parquet", index=False)

    log.info("Construindo features de par user-produto...")
    user_product_features = build_user_product_features(
        interactions=interactions,
        orders=orders,
    )
    user_product_features.to_parquet(
        FEATURES_DIR / "user_product_features.parquet", index=False
    )

    log.info("Criando splits train/val/test (seed=%d)...", settings.seed)
    train, val, test = split_dataset(
        user_features=user_features,
        product_features=product_features,
        user_product_features=user_product_features,
        interactions=interactions,
        seed=settings.seed,
    )
    train.to_parquet(FEATURES_DIR / "train.parquet", index=False)
    val.to_parquet(FEATURES_DIR / "val.parquet", index=False)
    test.to_parquet(FEATURES_DIR / "test.parquet", index=False)

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

    log.info("Stage 2 concluído. %s", metrics)


if __name__ == "__main__":
    run()
