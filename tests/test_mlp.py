"""Testes do módulo de MLP."""

import torch

from models.mlp import RecommenderMLP


def test_forward_returns_tensor():
    """Verifica se o método forward retorna um tensor."""
    model = RecommenderMLP(n_users=10, n_products=20)
    user_ids = torch.tensor([0, 1, 2])
    product_ids = torch.tensor([0, 1, 2])
    output = model.forward(user_ids, product_ids)
    assert isinstance(output, torch.Tensor)


def test_forward_output_shape():
    """Valida se a saída do método forward tem o shape correto."""
    model = RecommenderMLP(n_users=10, n_products=20)
    user_ids = torch.tensor([0, 1, 2])
    product_ids = torch.tensor([0, 1, 2])
    output = model.forward(user_ids, product_ids)
    assert output.shape == torch.Size([3, 1])


def test_forward_with_different_batch_sizes():
    """Verifica se o método forward funciona com diferentes tamanhos de batch."""
    model = RecommenderMLP(n_users=10, n_products=20)
    for batch_size in [1, 5, 10]:
        user_ids = torch.randint(0, 10, (batch_size,))
        product_ids = torch.randint(0, 20, (batch_size,))
        output = model.forward(user_ids, product_ids)
        assert output.shape == torch.Size([batch_size, 1])
