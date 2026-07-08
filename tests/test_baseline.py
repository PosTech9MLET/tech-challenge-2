"""Testes para o módulo de baseline."""

import pandas as pd
import pytest

from src.models.baseline import PopularityBaseline
from src.models.factory import ModelFactory


@pytest.fixture
def interactions_df() -> pd.DataFrame:
    """DataFrame mínimo de interações para testes."""
    return pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "product_id": [101, 102, 101, 103],
            "reordered": [1, 0, 1, 1],
        }
    )


def test_create_baseline() -> None:
    """Verifica se ModelFactory cria um PopularityBaseline."""
    model = ModelFactory.create("baseline")
    assert isinstance(model, PopularityBaseline)


def test_fit_learns_top_products(interactions_df) -> None:
    """Valida se _top_products é preenchido após o fit."""
    model = PopularityBaseline()
    model.fit(interactions_df)
    assert hasattr(model, "_top_products")
    assert len(model._top_products) > 0


def test_predict_returns_dict(interactions_df) -> None:
    """Verifica se o método predict retorna um dicionário."""
    model = PopularityBaseline()
    model.fit(interactions_df)
    predictions = model.predict([1, 2])
    assert isinstance(predictions, dict)
    assert set(predictions.keys()) == {1, 2}


def test_predict_respects_k(interactions_df) -> None:
    """Valida se o método predict respeita o parâmetro k."""
    model = PopularityBaseline()
    model.fit(interactions_df)
    predictions = model.predict([1, 2], k=1)
    for user_id in predictions:
        assert len(predictions[user_id]) == 1


def test_evaluate_returns_metrics(interactions_df) -> None:
    """Verifica se o método evaluate retorna as 4 métricas esperadas."""
    model = PopularityBaseline()
    model.fit(interactions_df)
    metrics = model.evaluate(interactions_df)
    assert isinstance(metrics, dict)
    assert "precision_at_k" in metrics
    assert "recall_at_k" in metrics
    assert "ndcg_at_k" in metrics
    assert "hit_rate_at_k" in metrics
