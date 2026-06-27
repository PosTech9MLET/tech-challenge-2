"""Stage 3 do pipeline DVC — treinamento do baseline e do MLP."""

import json
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient

from configs.settings import settings
from src.models.baseline import PopularityBaseline
from src.models.factory import ModelFactory
from src.models.mlp import RecommenderMLP
from src.training.trainer import MLPTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
PARAMS_FILE = Path("params.yaml")
MODEL_NAME = "recommender-mlp"


def load_params() -> dict:
    """Carrega os hiperparâmetros do params.yaml.

    Returns:
        Dicionário com os parâmetros de treino e modelo.
    """
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def setup_mlflow() -> None:
    """Configura o tracking URI e o experimento do MLflow."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("tech-challenge-recommendation")


def train_baseline(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[PopularityBaseline, dict]:
    """Treina e avalia o PopularityBaseline, logando no MLflow.

    Args:
        train: DataFrame de treino.
        val: DataFrame de validação.

    Returns:
        Tupla (baseline treinado, métricas de validação).
    """
    log.info("Treinando PopularityBaseline...")
    with mlflow.start_run(run_name="popularity-baseline"):
        baseline: PopularityBaseline = ModelFactory.create("baseline")
        baseline.fit(train)
        metrics = baseline.evaluate(val, k=10)
        mlflow.log_params({"model_type": "PopularityBaseline", "k": 10})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(baseline, artifact_path="baseline_model")
        log.info("Baseline val metrics: %s", metrics)
    return baseline, metrics


def build_mlp_params(params: dict) -> dict:
    """Monta o dicionário de hiperparâmetros do MLP.

    Combina valores do params.yaml com os do .env (settings), que
    têm precedência para seed e early_stopping_patience.

    Args:
        params: Dicionário carregado do params.yaml.

    Returns:
        Dicionário plano com todos os hiperparâmetros do MLP.
    """
    return {
        "embedding_dim": params["model"]["embedding_dim"],
        "hidden_layers": params["model"]["hidden_layers"],
        "dropout": params["model"]["dropout"],
        "seed": settings.seed,
        "epochs": params["train"]["epochs"],
        "batch_size": params["train"]["batch_size"],
        "lr": params["train"]["lr"],
        "early_stopping_patience": settings.early_stopping_patience,
    }


def _build_model_and_trainer(
    train: pd.DataFrame,
    mlp_params: dict,
) -> tuple[RecommenderMLP, MLPTrainer]:
    """Instancia o RecommenderMLP e o MLPTrainer a partir dos params.

    Args:
        train: DataFrame de treino, usado para inferir n_users e
            n_products a partir dos IDs codificados.
        mlp_params: Hiperparâmetros do modelo e do treino.

    Returns:
        Tupla (model, trainer) ainda não treinados.
    """
    n_users = int(train["user_id_enc"].max()) + 1
    n_products = int(train["product_id_enc"].max()) + 1
    model: RecommenderMLP = ModelFactory.create(
        "mlp",
        n_users=n_users,
        n_products=n_products,
        embedding_dim=mlp_params["embedding_dim"],
        hidden_layers=mlp_params["hidden_layers"],
        dropout=mlp_params["dropout"],
    )
    trainer = MLPTrainer(
        model=model,
        lr=mlp_params["lr"],
        batch_size=mlp_params["batch_size"],
        epochs=mlp_params["epochs"],
        patience=mlp_params["early_stopping_patience"],
        seed=mlp_params["seed"],
    )
    return model, trainer


def train_mlp(
    train: pd.DataFrame,
    val: pd.DataFrame,
    mlp_params: dict,
) -> tuple[MLPTrainer, dict, str]:
    """Treina o RecommenderMLP e loga tudo no MLflow.

    Args:
        train: DataFrame de treino.
        val: DataFrame de validação.
        mlp_params: Hiperparâmetros do modelo e do treino.

    Returns:
        Tupla (trainer treinado, métricas de validação, run_id).
    """
    log.info("Treinando RecommenderMLP (seed=%d)...", mlp_params["seed"])
    with mlflow.start_run(run_name="mlp-embedding") as run:
        mlflow.log_params(mlp_params)
        model, trainer = _build_model_and_trainer(train, mlp_params)
        metrics = trainer.fit(train=train, val=val)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, artifact_path="mlp_model")
        log.info("MLP val metrics: %s", metrics)
        run_id = run.info.run_id
    return trainer, metrics, run_id


def _set_alias_staging(
    client: MlflowClient,
    version: str,
) -> None:
    """Define o alias 'challenger' na versão em avaliação.

    Na API moderna do MLflow (≥2.9), aliases substituem os stages
    deprecated. 'challenger' equivale ao antigo 'Staging'.

    Args:
        client: Cliente MLflow autenticado.
        version: Versão do modelo no Registry.
    """
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    log.info("Alias 'challenger' definido para '%s' v%s.", MODEL_NAME, version)


def _set_alias_production(
    client: MlflowClient,
    version: str,
) -> None:
    """Promove a versão para produção definindo o alias 'champion'.

    Na API moderna do MLflow (≥2.9), aliases substituem os stages
    deprecated. 'champion' equivale ao antigo 'Production'.

    Args:
        client: Cliente MLflow autenticado.
        version: Versão do modelo no Registry.
    """
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    log.info("Alias 'champion' definido para '%s' v%s.", MODEL_NAME, version)


def register_model(run_id: str, val_metrics: dict) -> None:
    """Registra o MLP no Model Registry seguindo o fluxo challenger → champion.

    Usa a API moderna de aliases do MLflow (≥2.9) em substituição aos
    stages deprecated. O fluxo challenger → champion equivale ao
    antigo Staging → Production.

    Args:
        run_id: ID do run do MLflow onde o modelo foi logado.
        val_metrics: Métricas de validação para registrar como tags.
    """
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/mlp_model"

    log.info("Registrando modelo no MLflow Model Registry...")
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    version = registered.version

    client.set_registered_model_tag(MODEL_NAME, "framework", "pytorch")
    client.set_registered_model_tag(MODEL_NAME, "dataset", "instacart")
    for metric, value in val_metrics.items():
        client.set_model_version_tag(MODEL_NAME, version, metric, str(value))

    _set_alias_staging(client, version)
    _set_alias_production(client, version)


def save_artifacts(
    baseline: PopularityBaseline,
    trainer: MLPTrainer,
    baseline_metrics: dict,
    mlp_metrics: dict,
) -> None:
    """Persiste os artefatos do treino em disco.

    Args:
        baseline: Modelo baseline treinado.
        trainer: Trainer com o MLP treinado.
        baseline_metrics: Métricas de validação do baseline.
        mlp_metrics: Métricas de validação do MLP.
    """
    with open(MODELS_DIR / "baseline.pkl", "wb") as f:
        pickle.dump(baseline, f)
    trainer.save(MODELS_DIR / "mlp_best.pt")
    with open(MODELS_DIR / "train_metrics.json", "w") as f:
        json.dump({"baseline": baseline_metrics, "mlp": mlp_metrics}, f, indent=2)


def run() -> None:
    """Treina baseline e MLP, loga tudo no MLflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params = load_params()
    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")

    setup_mlflow()

    baseline, baseline_metrics = train_baseline(train, val)
    mlp_params = build_mlp_params(params)
    trainer, mlp_metrics, run_id = train_mlp(train, val, mlp_params)

    save_artifacts(baseline, trainer, baseline_metrics, mlp_metrics)
    register_model(run_id, mlp_metrics)

    log.info("Stage 3 concluído.")


if __name__ == "__main__":
    run()
