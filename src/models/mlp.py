"""Modelo MLP embedding-based para recomendação de produtos."""

import torch
import torch.nn as nn


def _build_hidden_layers(
    input_dim: int,
    hidden_layers: list[int],
    dropout: float,
) -> list[nn.Module]:
    """Constrói a sequência de camadas densas do MLP.

    Args:
        input_dim: Dimensão de entrada (embeddings concatenados).
        hidden_layers: Tamanho de cada camada oculta.
        dropout: Taxa de dropout aplicada após cada camada.

    Returns:
        Lista de módulos PyTorch (Linear, ReLU, Dropout) intercalados,
        terminando em uma camada de saída de dimensão 1.
    """
    layers: list[nn.Module] = []
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        input_dim = hidden_dim
    layers.append(nn.Linear(input_dim, 1))
    return layers


class RecommenderMLP(nn.Module):
    """MLP com embeddings de usuário e produto para predição de reordem.

    Arquitetura: Embedding(user) + Embedding(product) é concatenado e
    passado por um MLP que termina em um único logit (probabilidade
    de recompra, após sigmoid).
    """

    def __init__(
        self,
        n_users: int,
        n_products: int,
        embedding_dim: int = 32,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        """Inicializa o modelo.

        Args:
            n_users: Número de usuários únicos (tamanho do embedding).
            n_products: Número de produtos únicos (tamanho do embedding).
            embedding_dim: Dimensão dos vetores de embedding.
            hidden_layers: Tamanho de cada camada oculta do MLP.
            dropout: Taxa de dropout entre as camadas densas.
        """
        super().__init__()
        hidden_layers = hidden_layers or [256, 128, 64]

        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.product_emb = nn.Embedding(n_products, embedding_dim)

        layers = _build_hidden_layers(
            input_dim=embedding_dim * 2,
            hidden_layers=hidden_layers,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        user_ids: torch.Tensor,
        product_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Executa o forward pass do modelo.

        Args:
            user_ids: Tensor de índices de usuário, shape (batch,).
            product_ids: Tensor de índices de produto, shape (batch,).

        Returns:
            Tensor de logits, shape (batch, 1). Aplicar sigmoid para
            obter a probabilidade de recompra.
        """
        user_vec = self.user_emb(user_ids)
        product_vec = self.product_emb(product_ids)
        combined = torch.cat([user_vec, product_vec], dim=-1)
        return self.mlp(combined)
