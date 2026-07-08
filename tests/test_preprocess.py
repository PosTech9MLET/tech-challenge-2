"""Testes de preprocessamento."""

# Importação das bibliotecas necessárias
import pandas as pd
import pytest

from src.features.preprocess import load_orders


# cria a fixture orders_df
@pytest.fixture
def orders_df() -> pd.DataFrame:
    """Cria um DataFrame de pedidos para os testes."""
    data = {
        "user_id": [1, 1, 2, 2, 3],
        "order_id": [101, 102, 201, 202, 301],
        "days_since_prior_order": [5, 10, 3, 7, None],
        "order_hour_of_day": [10, 14, 9, 20, 15],
        "order_dow": [1, 2, 3, 4, 5],
        "order_number": [1, 2, 1, 2, 1],
    }
    return pd.DataFrame(data)


def test_is_first_order_flag(orders_df, tmp_path) -> None:
    """Teste para verificar se a flag is_first_order é criada corretamente."""
    # arrange
    path = tmp_path / "orders.csv"
    orders_df.to_csv(path, index=False)

    # act
    result = load_orders(str(path))

    # asserts
    assert result.loc[result["user_id"] == 3, "is_first_order"].values[0] == 1
    assert result.loc[result["user_id"] == 1, "is_first_order"].values[0] == 0


def test_days_not_first_order(orders_df, tmp_path) -> None:
    """Verifica se days_since_prior_order é mantido para pedidos não iniciais."""
    # arrange
    path = tmp_path / "orders.csv"
    orders_df.to_csv(path, index=False)

    # act
    result = load_orders(str(path))

    # asserts
    assert result.loc[result["user_id"] == 1, "is_first_order"].values[0] == 0
    assert result.loc[result["user_id"] == 1, "days_since_prior_order"].values[0] == 5


def test_fillna_with_median(orders_df, tmp_path) -> None:
    """Verifica se NaN em days_since_prior_order é preenchido com a mediana."""
    # arrange
    path = tmp_path / "orders.csv"
    orders_df.to_csv(path, index=False)

    # act
    result = load_orders(str(path))

    # asserts
    assert result.loc[result["user_id"] == 3, "days_since_prior_order"].values[0] == 6.0


def test_missing_column_raises(tmp_path) -> None:
    """Verifica se ValueError é levantado quando coluna obrigatória está ausente."""
    df = pd.DataFrame({"user_id": [1], "order_id": [1]})
    path = tmp_path / "orders.csv"
    df.to_csv(path, index=False)

    with pytest.raises((ValueError, KeyError)):
        load_orders(str(path))
