"""Lógica de treinamento do RecommenderMLP com early stopping."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.baseline import compute_ranking_metrics
from src.models.mlp import RecommenderMLP

log = logging.getLogger(__name__)


def _make_loader(df: pd.DataFrame, batch_size: int) -> DataLoader:
    """Cria um DataLoader de tensores a partir de um DataFrame.

    Args:
        df: DataFrame com colunas user_id_enc, product_id_enc, reordered.
        batch_size: Tamanho do batch.

    Returns:
        DataLoader configurado com shuffle ativado.
    """
    users = torch.tensor(df["user_id_enc"].values, dtype=torch.long)
    products = torch.tensor(df["product_id_enc"].values, dtype=torch.long)
    labels = torch.tensor(df["reordered"].values, dtype=torch.float32)
    dataset = TensorDataset(users, products, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def _run_train_epoch(
    model: RecommenderMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Executa uma época de treino e retorna a loss média.

    Args:
        model: Modelo a ser treinado.
        loader: DataLoader de treino.
        optimizer: Otimizador configurado.
        criterion: Função de perda.
        device: Dispositivo de execução (CPU).

    Returns:
        Loss média da época.
    """
    model.train()
    total_loss = 0.0
    for user_ids, product_ids, labels in loader:
        user_ids = user_ids.to(device)
        product_ids = product_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(user_ids, product_ids).squeeze()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _run_eval_loss(
    model: RecommenderMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Calcula a loss média em um DataLoader, sem gradiente.

    Args:
        model: Modelo a ser avaliado.
        loader: DataLoader de validação.
        criterion: Função de perda.
        device: Dispositivo de execução (CPU).

    Returns:
        Loss média no conjunto avaliado.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for user_ids, product_ids, labels in loader:
            user_ids = user_ids.to(device)
            product_ids = product_ids.to(device)
            labels = labels.to(device)
            logits = model(user_ids, product_ids).squeeze()
            total_loss += criterion(logits, labels).item()
    return total_loss / len(loader)


def _predict_scores(
    model: RecommenderMLP,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula a probabilidade de recompra para cada linha do teste.

    Args:
        model: Modelo treinado.
        test: DataFrame de teste com user_id_enc e product_id_enc.

    Returns:
        Cópia do DataFrame de teste com a coluna 'score' adicionada.
    """
    model.eval()
    user_ids = torch.tensor(test["user_id_enc"].values, dtype=torch.long)
    product_ids = torch.tensor(test["product_id_enc"].values, dtype=torch.long)

    with torch.no_grad():
        logits = model(user_ids, product_ids).squeeze()
        scores = torch.sigmoid(logits).numpy()

    scored = test.copy()
    scored["score"] = scores
    return scored


def _build_recommendations(
    scored: pd.DataFrame,
    k: int,
) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
    """Monta as recomendações top-k e a verdade de fato por usuário.

    Args:
        scored: DataFrame de teste com a coluna 'score' já calculada.
        k: Tamanho do ranking de recomendação.

    Returns:
        Tupla (recommendations, ground_truth), ambos indexados por
        user_id.
    """
    recommendations: dict[int, list[int]] = {}
    ground_truth: dict[int, set[int]] = {}

    for user_id, group in scored.groupby("user_id"):
        top_k = group.nlargest(k, "score")["product_id"].tolist()
        recommendations[user_id] = top_k
        bought = group[group["reordered"] == 1]["product_id"]
        ground_truth[user_id] = set(bought.tolist())

    return recommendations, ground_truth


class _EarlyStopper:
    """Controla a parada antecipada do treino com base na val_loss."""

    def __init__(self, patience: int) -> None:
        """Inicializa o controlador.

        Args:
            patience: Épocas sem melhora toleradas antes de parar.
        """
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict = {}

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Avalia a época atual e decide se deve parar o treino.

        Args:
            val_loss: Loss de validação da época atual.
            model: Modelo sendo treinado, para snapshot dos pesos.

        Returns:
            True se o treino deve parar (patience esgotada).
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience


class MLPTrainer:
    """Treina o RecommenderMLP com early stopping baseado em val_loss."""

    def __init__(
        self,
        model: RecommenderMLP,
        lr: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 30,
        patience: int = 5,
        seed: int = 42,
    ) -> None:
        """Inicializa o trainer.

        Args:
            model: Modelo a ser treinado.
            lr: Taxa de aprendizado do otimizador Adam.
            batch_size: Tamanho do batch.
            epochs: Número máximo de épocas de treino.
            patience: Épocas sem melhora antes do early stopping.
            seed: Semente aleatória para reprodutibilidade.
        """
        torch.manual_seed(seed)
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seed = seed
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> dict[str, float]:
        """Treina o modelo com early stopping baseado na loss de validação.

        Args:
            train: DataFrame de treino.
            val: DataFrame de validação.

        Returns:
            Dicionário com a métrica val_loss da melhor época encontrada.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        train_loader = _make_loader(train, self.batch_size)
        val_loader = _make_loader(val, self.batch_size)
        stopper = _EarlyStopper(self.patience)

        for epoch in range(1, self.epochs + 1):
            should_stop = self._run_epoch(
                epoch, train_loader, val_loader, optimizer, criterion, stopper
            )
            if should_stop:
                log.info("Early stopping na época %d.", epoch)
                break

        self.model.load_state_dict(stopper.best_state)
        return {"val_loss": stopper.best_loss}

    def _run_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        stopper: _EarlyStopper,
    ) -> bool:
        """Executa uma época de treino e validação.

        Args:
            epoch: Número da época atual (1-indexed).
            train_loader: DataLoader de treino.
            val_loader: DataLoader de validação.
            optimizer: Otimizador configurado.
            criterion: Função de perda.
            stopper: Controlador de early stopping.

        Returns:
            True se o early stopping deve interromper o treino.
        """
        train_loss = _run_train_epoch(
            self.model, train_loader, optimizer, criterion, self.device
        )
        val_loss = _run_eval_loss(self.model, val_loader, criterion, self.device)
        log.info(
            "Epoch %d/%d - train_loss=%.4f val_loss=%.4f",
            epoch,
            self.epochs,
            train_loss,
            val_loss,
        )
        return stopper.step(val_loss, self.model)

    def evaluate(self, test: pd.DataFrame, k: int = 10) -> dict[str, float]:
        """Avalia o modelo no teste com métricas de ranking.

        Args:
            test: DataFrame de teste.
            k: Tamanho da lista de recomendação avaliada.

        Returns:
            Dicionário com precision_at_k, recall_at_k, ndcg_at_k e
            hit_rate_at_k.
        """
        scored = _predict_scores(self.model, test)
        recommendations, ground_truth = _build_recommendations(scored, k)
        return compute_ranking_metrics(recommendations, ground_truth, k)

    def save(self, path: str | Path) -> None:
        """Salva o state_dict do modelo em disco.

        Args:
            path: Caminho do arquivo de checkpoint (.pt).
        """
        torch.save(self.model.state_dict(), path)
        log.info("Modelo salvo em %s", path)

    @classmethod
    def load(cls, path: str | Path, model: RecommenderMLP) -> MLPTrainer:
        """Carrega os pesos de um checkpoint em um modelo já instanciado.

        Args:
            path: Caminho do arquivo de checkpoint (.pt).
            model: Instância de RecommenderMLP com a mesma arquitetura
                usada no treino original.

        Returns:
            Instância de MLPTrainer com o modelo carregado e pronto
            para avaliação.
        """
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        trainer = cls(model=model)
        log.info("Modelo carregado de %s", path)
        return trainer
