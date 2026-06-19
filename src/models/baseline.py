"""Baseline de recomendação por popularidade global."""

import numpy as np
import pandas as pd


def compute_ranking_metrics(
    recommendations: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    k: int,
) -> dict[str, float]:
    """Calcula Precision@K, Recall@K, NDCG@K e HR@K.

    Args:
        recommendations: Dicionário {user_id: [produtos recomendados]}.
        ground_truth: Dicionário {user_id: {produtos relevantes}}.
        k: Tamanho do ranking avaliado.

    Returns:
        Dicionário com as 4 métricas calculadas e médias entre usuários.
    """
    precisions, recalls, ndcgs, hits = [], [], [], []

    for user, recs in recommendations.items():
        relevant = ground_truth.get(user, set())
        if not relevant:
            continue

        scored = _score_recommendation(recs[:k], relevant)
        precisions.append(scored["precision"])
        recalls.append(scored["recall"])
        ndcgs.append(scored["ndcg"])
        hits.append(scored["hit"])

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate_at_k": float(np.mean(hits)) if hits else 0.0,
    }


def _score_recommendation(
    recs_k: list[int],
    relevant: set[int],
) -> dict[str, float]:
    """Calcula as métricas de ranking para um único usuário.

    Args:
        recs_k: Lista dos top-k produtos recomendados ao usuário.
        relevant: Conjunto de produtos relevantes (verdade de fato).

    Returns:
        Dicionário com precision, recall, ndcg e hit para esse usuário.
    """
    k = len(recs_k)
    hits_k = [1 if r in relevant else 0 for r in recs_k]

    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits_k))
    ideal = sorted(hits_k, reverse=True)
    idcg = sum(h / np.log2(i + 2) for i, h in enumerate(ideal)) or 1.0

    return {
        "precision": sum(hits_k) / k if k else 0.0,
        "recall": sum(hits_k) / len(relevant),
        "ndcg": dcg / idcg,
        "hit": float(sum(hits_k) > 0),
    }


class PopularityBaseline:
    """Recomenda os produtos mais populares no histórico de compras.

    Serve como referência de comparação para o RecommenderMLP: não
    personaliza nada, apenas recomenda os mesmos itens mais populares
    para todos os usuários.
    """

    def __init__(self, k: int = 10) -> None:
        """Inicializa o baseline.

        Args:
            k: Número de itens a recomendar por padrão.
        """
        self.k = k
        self._top_products: list[int] = []

    def fit(self, train: pd.DataFrame) -> "PopularityBaseline":
        """Aprende os produtos mais recomprados no treino.

        Args:
            train: DataFrame com colunas product_id e reordered.

        Returns:
            Self, para permitir encadeamento de chamadas.
        """
        counts = (
            train.groupby("product_id")["reordered"].sum().sort_values(ascending=False)
        )
        self._top_products = counts.index.tolist()
        return self

    def predict(
        self,
        users: list[int],
        k: int | None = None,
    ) -> dict[int, list[int]]:
        """Retorna os top-k produtos mais populares para cada usuário.

        Args:
            users: Lista de user_ids a recomendar.
            k: Tamanho da lista de recomendação (usa self.k se None).

        Returns:
            Dicionário {user_id: [product_id, ...]}.
        """
        k = k or self.k
        top_k = self._top_products[:k]
        return {u: top_k for u in users}

    def evaluate(self, data: pd.DataFrame, k: int = 10) -> dict[str, float]:
        """Calcula as métricas de ranking no conjunto de dados informado.

        Args:
            data: DataFrame com colunas user_id, product_id e reordered.
            k: Tamanho da lista de recomendação avaliada.

        Returns:
            Dicionário com precision_at_k, recall_at_k, ndcg_at_k e
            hit_rate_at_k.
        """
        users = data["user_id"].unique().tolist()
        recommendations = self.predict(users, k=k)
        ground_truth = _build_ground_truth(data)
        return compute_ranking_metrics(recommendations, ground_truth, k)


def _build_ground_truth(data: pd.DataFrame) -> dict[int, set[int]]:
    """Constrói o dicionário de produtos relevantes por usuário.

    Args:
        data: DataFrame com colunas user_id, product_id e reordered.

    Returns:
        Dicionário {user_id: {product_ids efetivamente recomprados}}.
    """
    relevant = data[data["reordered"] == 1]
    return relevant.groupby("user_id")["product_id"].apply(set).to_dict()
