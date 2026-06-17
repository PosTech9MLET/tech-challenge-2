"""Stage 3 do pipeline DVC — treinamento do baseline e do MLP."""

import json
import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from src.models.baseline import PopularityBaseline
from src.models.mlp import RecommenderMLP
from src.training.trainer import MLPTrainer

from configs.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
PARAMS_FILE = Path("params.yaml")


def load_params() -> dict:
    """Carrega os hiperparâmetros do params.yaml.

    Returns:
        Dicionário com os parâmetros de treino e modelo.
    """
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def run() -> None:
    """Treina baseline e MLP, loga tudo no MLflow."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    params = load_params()
    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    val = pd.read_parquet(FEATURES_DIR / "val.parquet")

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    mlflow.set_experiment("tech-challenge-recommendation")

    # --- Baseline ---
    log.info("Treinando PopularityBaseline...")
    with mlflow.start_run(run_name="popularity-baseline"):
        baseline = PopularityBaseline()
        baseline.fit(train)
        baseline_metrics = baseline.evaluate(val, k=10)
        mlflow.log_params({"model_type": "PopularityBaseline", "k": 10})
        mlflow.log_metrics(baseline_metrics)
        log.info("Baseline val metrics: %s", baseline_metrics)

    with open(MODELS_DIR / "baseline.pkl", "wb") as f:
        pickle.dump(baseline, f)

    # --- MLP Embedding-based ---
    log.info("Treinando RecommenderMLP (seed=%d)...", settings.seed)
    mlp_params = {
        "embedding_dim": params["model"]["embedding_dim"],
        "hidden_layers": params["model"]["hidden_layers"],
        "dropout": params["model"]["dropout"],
        "seed": settings.seed,
        "epochs": params["train"]["epochs"],
        "batch_size": params["train"]["batch_size"],
        "lr": params["train"]["lr"],
        "early_stopping_patience": settings.early_stopping_patience,
    }

    with mlflow.start_run(run_name="mlp-embedding"):
        mlflow.log_params(mlp_params)

        n_users = int(train["user_id_enc"].max()) + 1
        n_products = int(train["product_id_enc"].max()) + 1

        model = RecommenderMLP(
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
        mlp_metrics = trainer.fit(train=train, val=val)
        mlflow.log_metrics(mlp_metrics)
        log.info("MLP val metrics: %s", mlp_metrics)

    trainer.save(MODELS_DIR / "mlp_best.pt")

    with open(MODELS_DIR / "train_metrics.json", "w") as f:
        json.dump({"baseline": baseline_metrics, "mlp": mlp_metrics}, f, indent=2)

    log.info("Stage 3 concluído.")


if __name__ == "__main__":
    run()
