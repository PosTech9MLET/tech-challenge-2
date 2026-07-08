"""Engenharia de feature."""

import pandas as pd


def build_user_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Cria features relacionadas aos usuários.

    Args:
        orders: DataFrame contendo os dados de pedidos.

    Returns:
        DataFrame com as features de usuário.
    """
    required_cols = [
        "user_id",
        "order_id",
        "days_since_prior_order",
        "order_hour_of_day",
        "order_dow",
        "order_number",
    ]

    missing = [c for c in required_cols if c not in orders.columns]
    if missing:
        raise ValueError(f"Colunas faltando: {missing}")

    user_features = (
        orders.groupby("user_id")
        .agg(
            total_orders=("order_id", "nunique"),
            avg_days_since_prior=("days_since_prior_order", "mean"),
            std_days_since_prior=("days_since_prior_order", "std"),
            median_hour=("order_hour_of_day", "median"),
            preferred_dow=("order_dow", lambda x: x.mode()[0]),
        )
        .reset_index()
    )

    return user_features


def build_product_features(
    order_products_prior: pd.DataFrame,
    products: pd.DataFrame,
    aisles: pd.DataFrame,
    departments: pd.DataFrame,
) -> pd.DataFrame:
    """Cria features relacionadas aos produtos.

    Args:
        order_products_prior: DataFrame com os produtos dos pedidos anteriores.
        products: DataFrame com informações dos produtos.
        aisles: DataFrame com informações dos corredores.
        departments: DataFrame com informações dos departamentos.

    Returns:
        DataFrame com as features de produto.
    """
    required_cols = ["product_id", "order_id", "add_to_cart_order", "reordered"]

    missing = [c for c in required_cols if c not in order_products_prior.columns]
    if missing:
        raise ValueError(f"Colunas faltando: {missing}")

    product_features = (
        order_products_prior.groupby("product_id")
        .agg(
            product_reorder_rate=("reordered", "mean"),
            total_orders_product=("order_id", "nunique"),
            avg_cart_position=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )

    # enriquece com dados do catálogo
    product_features = product_features.merge(products, on="product_id", how="left")
    product_features = product_features.merge(aisles, on="aisle_id", how="left")
    product_features = product_features.merge(
        departments, on="department_id", how="left"
    )

    # cria flag orgânico
    product_features["is_organic"] = (
        product_features["product_name"].str.contains("organic", case=False)
    ).astype(int)

    return product_features
