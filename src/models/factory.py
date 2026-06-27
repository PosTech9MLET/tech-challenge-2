"""Factory Pattern para criação de modelos de recomendação."""

from typing import Protocol

import pandas as pd


class RecommenderModel(Protocol):
    """Interface comum para modelos de recomendação.

    Define o contrato que todos os modelos devem seguir,
    permitindo que o Factory retorne qualquer implementação
    de forma transparente para o chamador.
    """

    def fit(self, train: pd.DataFrame) -> "RecommenderModel":
        """Treina o modelo com os dados fornecidos.

        Args:
            train: DataFrame de treino.

        Returns:
            Self, para encadeamento de chamadas.
        """
        ...

    def evaluate(self, data: pd.DataFrame, k: int) -> dict[str, float]:
        """Avalia o modelo e retorna métricas de ranking.

        Args:
            data: DataFrame de avaliação.
            k: Tamanho da lista de recomendação.

        Returns:
            Dicionário com as métricas calculadas.
        """
        ...


class ModelFactory:
    """Factory para instanciar modelos de recomendação por nome.

    Centraliza a criação de modelos, desacoplando o código cliente
    das implementações concretas. Para adicionar um novo modelo,
    basta registrá-lo no dicionário _registry sem alterar o chamador.

    Example:
        >>> model = ModelFactory.create("baseline")
        >>> model.fit(train_df)
    """

    @staticmethod
    def create(model_type: str, **kwargs) -> RecommenderModel:
        """Instancia e retorna o modelo correspondente ao tipo informado.

        Args:
            model_type: Identificador do modelo. Valores aceitos:
                'baseline' para PopularityBaseline,
                'mlp' para RecommenderMLP.
            **kwargs: Argumentos repassados ao construtor do modelo.

        Returns:
            Instância do modelo solicitado.

        Raises:
            ValueError: Se o model_type não for reconhecido.
        """
        registry = ModelFactory._build_registry()
        if model_type not in registry:
            valid = list(registry.keys())
            raise ValueError(
                f"Modelo '{model_type}' não reconhecido. Opções válidas: {valid}"
            )
        return registry[model_type](**kwargs)

    @staticmethod
    def _build_registry() -> dict:
        """Constrói o dicionário de modelos disponíveis.

        Importações locais evitam dependências circulares e tornam
        o registro lazy — só carrega o que for necessário.

        Returns:
            Dicionário {nome: classe} dos modelos registrados.
        """
        from src.models.baseline import PopularityBaseline
        from src.models.mlp import RecommenderMLP

        return {
            "baseline": PopularityBaseline,
            "mlp": RecommenderMLP,
        }

    @staticmethod
    def available_models() -> list[str]:
        """Lista os modelos disponíveis no registry.

        Returns:
            Lista com os nomes dos modelos registrados.
        """
        return list(ModelFactory._build_registry().keys())
