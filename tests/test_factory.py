"""Teste para o módulo de Factory."""

import pytest

from src.models.baseline import PopularityBaseline
from src.models.factory import ModelFactory
from src.models.mlp import RecommenderMLP


def test_create_baseline():
    """Verifica se ModelFactory cria um PopularityBaseline."""
    model = ModelFactory.create("baseline")
    assert isinstance(model, PopularityBaseline)


def test_create_mlp():
    """Verifica se ModelFactory cria um RecommenderMLP."""
    model = ModelFactory.create("mlp", n_users=10, n_products=20)
    assert isinstance(model, RecommenderMLP)


def test_create_invalid_raises():
    """Verifica se ModelFactory lança ValueError para modelo inválido."""
    with pytest.raises(ValueError):
        ModelFactory.create("invalid_model")


def test_available_models():
    """Verifica se ModelFactory retorna os modelos disponíveis."""
    available = ModelFactory.available_models()
    assert "baseline" in available
    assert "mlp" in available
