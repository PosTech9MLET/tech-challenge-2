"""Construção de features para o modelo de recomendação do Instacart."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _aggregate_order_stats(orders: pd.DataFrame) -> pd.DataFrame:
    """Agrega estatísticas de pedidos por usuário.

    Args:
        orders: Pedidos limpos com metadados temporais.

    Returns:
        DataFrame com total de pedidos, intervalo médio e horário médio.
    """
    return (
        orders.groupby("user_id")
        .agg(
            total_orders=("order_id", "nunique"),
            avg_days_since_prior=("days_since_prior_order", "mean"),
            std_days_since_prior=("days_since_prior_order", "std"),
            avg_hour=("order_hour_of_day", "mean"),
        )
        .reset_index()
    )


def _aggregate_user_reorder_rate(prior: pd.DataFrame) -> pd.DataFrame:
    """Calcula a taxa média de recompra por usuário.

    Args:
        prior: Interações do split histórico (prior).

    Returns:
        DataFrame com user_id e avg_reorder_rate.
    """
    return (
        prior.groupby("user_id")
        .agg(avg_reorder_rate=("reordered", "mean"))
        .reset_index()
    )


def build_user_features(
    orders: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega features de comportamento por usuário.

    Args:
        orders: Pedidos limpos com metadados temporais.
        interactions: Tabela unificada de interações.

    Returns:
        DataFrame com uma linha por usuário e features agregadas.
    """
    prior = interactions[interactions["split"] == "prior"]

    order_stats = _aggregate_order_stats(orders)
    reorder_stats = _aggregate_user_reorder_rate(prior)

    user_features = order_stats.merge(reorder_stats, on="user_id", how="left")
    fill_cols = ["std_days_since_prior", "avg_reorder_rate"]
    user_features[fill_cols] = user_features[fill_cols].fillna(0)
    return user_features


def _aggregate_product_stats(prior: pd.DataFrame) -> pd.DataFrame:
    """Agrega estatísticas de popularidade por produto.

    Args:
        prior: Interações do split histórico (prior).

    Returns:
        DataFrame com contagem de pedidos, taxa de recompra e
        posição média no carrinho, por produto.
    """
    return (
        prior.groupby("product_id")
        .agg(
            product_order_count=("order_id", "nunique"),
            product_reorder_rate=("reordered", "mean"),
            avg_cart_position=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )


def build_product_features(
    products_full: pd.DataFrame,
    interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega features de popularidade e fidelidade por produto.

    Args:
        products_full: Produtos enriquecidos com corredor e departamento.
        interactions: Tabela unificada de interações.

    Returns:
        DataFrame com uma linha por produto e features agregadas.
    """
    prior = interactions[interactions["split"] == "prior"]
    product_stats = _aggregate_product_stats(prior)

    base_cols = ["product_id", "aisle_id", "department_id", "is_organic"]
    product_features = products_full[base_cols].merge(
        product_stats, on="product_id", how="left"
    )

    fill_cols = ["product_order_count", "product_reorder_rate", "avg_cart_position"]
    product_features[fill_cols] = product_features[fill_cols].fillna(0)
    return product_features


def _aggregate_user_product_stats(prior: pd.DataFrame) -> pd.DataFrame:
    """Agrega estatísticas do par (usuário, produto).

    Args:
        prior: Interações do split histórico (prior).

    Returns:
        DataFrame com contagem de pedidos, taxa de recompra e posição
        média no carrinho, por par user-produto.
    """
    return (
        prior.groupby(["user_id", "product_id"])
        .agg(
            up_order_count=("order_id", "nunique"),
            up_avg_cart_position=("add_to_cart_order", "mean"),
        )
        .reset_index()
    )


def build_user_product_features(
    interactions: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula features do par (usuário, produto).

    Args:
        interactions: Tabela unificada de interações.
        orders: Pedidos limpos.

    Returns:
        DataFrame com features do par user-produto.
    """
    prior = interactions[interactions["split"] == "prior"]
    up_stats = _aggregate_user_product_stats(prior)
    prior_orders = orders[orders["eval_set"] == "prior"]

    total_orders = (
        prior_orders.groupby("user_id")["order_id"]
        .nunique()
        .rename("user_total_orders")
        .reset_index()
    )

    up_stats = up_stats.merge(total_orders, on="user_id", how="left")
    up_stats["up_order_rate"] = (
        up_stats["up_order_count"] / up_stats["user_total_orders"]
    )
    return up_stats


def _add_prefixed_features(
    dataset: pd.DataFrame,
    features: pd.DataFrame,
    key: str,
    prefix: str,
) -> pd.DataFrame:
    """Junta uma tabela de features ao dataset, prefixando suas colunas.

    Args:
        dataset: DataFrame base a ser enriquecido.
        features: Tabela de features a ser anexada.
        key: Nome da coluna chave usada no merge (ex: 'user_id').
        prefix: Prefixo aplicado às novas colunas (ex: 'u_').

    Returns:
        DataFrame resultante do merge com as colunas prefixadas.
    """
    cols = features.add_prefix(prefix)
    cols = cols.rename(columns={f"{prefix}{key}": key})
    return dataset.merge(cols, on=key, how="left")


def _merge_features(
    labels: pd.DataFrame,
    user_features: pd.DataFrame,
    product_features: pd.DataFrame,
    user_product_features: pd.DataFrame,
) -> pd.DataFrame:
    """Enriquece os rótulos com as 3 tabelas de features.

    Args:
        labels: DataFrame com user_id, product_id e reordered.
        user_features: Features agregadas de usuário.
        product_features: Features agregadas de produto.
        user_product_features: Features do par user-produto.

    Returns:
        DataFrame único com todas as features e o rótulo.
    """
    dataset = labels.merge(
        user_product_features, on=["user_id", "product_id"], how="left"
    )
    dataset = _add_prefixed_features(dataset, user_features, "user_id", "u_")
    dataset = _add_prefixed_features(dataset, product_features, "product_id", "p_")
    return dataset


def _encode_ids(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Cria índices inteiros sequenciais para usuários e produtos.

    Necessário porque as camadas de Embedding do PyTorch esperam
    índices contínuos começando em 0, não os IDs originais do Instacart.

    Args:
        dataset: DataFrame com colunas user_id e product_id.

    Returns:
        Tupla (dataset, user_encoder, product_encoder). O dataset
        ganha as colunas user_id_enc e product_id_enc, e os encoders
        são retornados para serem salvos e reutilizados na inferência.
    """
    user_enc = LabelEncoder()
    prod_enc = LabelEncoder()
    dataset["user_id_enc"] = user_enc.fit_transform(dataset["user_id"])
    dataset["product_id_enc"] = prod_enc.fit_transform(dataset["product_id"])
    return dataset, user_enc, prod_enc


def _split_users(users: np.ndarray, seed: int) -> tuple[set, set]:
    """Embaralha e divide usuários em conjuntos de treino e validação.

    Args:
        users: Array com os user_ids únicos do dataset.
        seed: Semente para reprodutibilidade do shuffle.

    Returns:
        Tupla (train_users, val_users) como conjuntos de user_ids.
        Os usuários restantes pertencem implicitamente ao teste.
    """
    rng = np.random.default_rng(seed)
    shuffled = users.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_users = set(shuffled[: int(n * 0.70)])
    val_users = set(shuffled[int(n * 0.70) : int(n * 0.85)])
    return train_users, val_users


def _sample_negatives_for_user(
    user_id: int,
    positive_products: set,
    popular_products: np.ndarray,
    n_negatives: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Sorteia produtos negativos populares para um usuário."""
    negatives = []
    attempts = 0
    max_attempts = n_negatives * 10

    while len(negatives) < n_negatives and attempts < max_attempts:
        product = rng.choice(popular_products)

        if product not in positive_products and product not in negatives:
            negatives.append(product)

        attempts += 1

    return pd.DataFrame(
        {
            "user_id": user_id,
            "product_id": negatives,
            "reordered": 0,
        }
    )


def add_negative_samples(
    labels: pd.DataFrame,
    popular_products: np.ndarray,
    seed: int,
    ratio: int = 20,
) -> pd.DataFrame:
    """Adiciona amostras negativas ao conjunto de rótulos.

    Sem isso, o modelo só é avaliado contra os itens que o usuário já
    decidiu comprar, o que infla artificialmente as métricas de
    ranking. Para cada usuário, sorteia produtos que ele não comprou
    no pedido de rótulo, na proporção indicada.

    Args:
        labels: DataFrame com user_id, product_id e reordered (só
            positivos, vindos do split 'train' nativo do Instacart).
        popular_products: Array com todos os product_ids do catálogo.
        seed: Semente para reprodutibilidade do sorteio.
        ratio: Número de negativos sorteados por positivo.

    Returns:
        DataFrame combinando positivos e negativos, com a coluna
        reordered indicando 1 (comprou) ou 0 (não comprou).
    """
    rng = np.random.default_rng(seed)
    positives_by_user = labels.groupby("user_id")["product_id"].apply(set)

    negative_frames = []
    for user_id, positive_products in positives_by_user.items():
        n_negatives = len(positive_products) * ratio
        negatives = _sample_negatives_for_user(
            user_id,
            positive_products,
            popular_products,
            n_negatives,
            rng,
        )
        negative_frames.append(negatives)

    negatives_df = pd.concat(negative_frames, ignore_index=True)
    return pd.concat([labels, negatives_df], ignore_index=True)


def _build_positive_labels(interactions: pd.DataFrame) -> pd.DataFrame:
    """Extrai os pares (usuário, produto) positivos do split 'train'.

    Args:
        interactions: Tabela de interações completa.

    Returns:
        DataFrame com user_id, product_id e reordered=1, um por item
        do próximo pedido real de cada usuário.
    """
    positives = interactions[interactions["split"] == "train"][
        ["user_id", "product_id", "reordered"]
    ].copy()
    positives["reordered"] = 1
    return positives


def split_dataset(
    user_features: pd.DataFrame,
    product_features: pd.DataFrame,
    user_product_features: pd.DataFrame,
    interactions: pd.DataFrame,
    seed: int = 42,
    negative_ratio: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, LabelEncoder, LabelEncoder]:
    """Gera splits por usuário (train 70% / val 15% / test 15%).

    Usa os itens do split nativo 'train' do Instacart como rótulos
    positivos (o próximo pedido do usuário), e adiciona amostras
    negativas para evitar avaliação inflada. Divide por user_id, não
    por linha, para evitar que o mesmo usuário apareça em mais de um
    conjunto (data leakage).

    Args:
        user_features: Features de usuário.
        product_features: Features de produto.
        user_product_features: Features do par user-produto.
        interactions: Tabela de interações completa.
        seed: Semente para reprodutibilidade do shuffle e do sampling.
        negative_ratio: Negativos sorteados por positivo.

    Returns:
        Tupla (train, val, test, user_encoder, product_encoder). Os
        encoders devem ser salvos para permitir inferência futura com
        os mesmos índices usados no treino.
    """
    positives = _build_positive_labels(interactions)
    popular_products = (
        interactions[interactions["split"] == "prior"]["product_id"]
        .value_counts()
        .head(2000)
        .index.to_numpy()
    )
    labels = add_negative_samples(positives, popular_products, seed, negative_ratio)

    dataset = _merge_features(
        labels, user_features, product_features, user_product_features
    )
    dataset, user_enc, prod_enc = _encode_ids(dataset)

    train_users, val_users = _split_users(dataset["user_id"].unique(), seed)
    is_train = dataset["user_id"].isin(train_users)
    is_val = dataset["user_id"].isin(val_users)

    train = dataset[is_train].reset_index(drop=True)
    val = dataset[is_val].reset_index(drop=True)
    test = dataset[~is_train & ~is_val].reset_index(drop=True)
    return train, val, test, user_enc, prod_enc
