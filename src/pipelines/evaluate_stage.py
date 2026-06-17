"""Stage 4 do pipeline DVC — avaliação final no conjunto de teste."""

import json
import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from src.models.baseline import PopularityBaseline
from src.training.trainer import MLPTrainer

from configs.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
PARAMS_FILE = Path("params.yaml")


def load_k_values() -> list[int]:
    """Carrega os valores de K do params.yaml.

    Returns:
        Lista de inteiros com os valores de K para avaliação.
    """
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)["evaluate"]["k_values"]


def run() -> None:
    """Avalia baseline e MLP no teste e registra métricas no MLflow."""
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    k_values = load_k_values()

    with open(MODELS_DIR / "baseline.pkl", "rb") as f:
        baseline: PopularityBaseline = pickle.load(f)

    trainer = MLPTrainer.load(MODELS_DIR / "mlp_best.pt")

    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    mlflow.set_experiment("tech-challenge-recommendation")

    all_metrics: dict = {}

    # --- Avalia baseline ---
    log.info("Avaliando PopularityBaseline no conjunto de teste...")
    with mlflow.start_run(run_name="eval-baseline"):
        for k in k_values:
            metrics = baseline.evaluate(test, k=k)
            mlflow.log_metrics({f"{key}_k{k}": v for key, v in metrics.items()})
            all_metrics[f"baseline_k{k}"] = metrics
            log.info("Baseline @%d: %s", k, metrics)

    # --- Avalia MLP ---
    log.info("Avaliando RecommenderMLP no conjunto de teste...")
    with mlflow.start_run(run_name="eval-mlp"):
        for k in k_values:
            metrics = trainer.evaluate(test, k=k)
            mlflow.log_metrics({f"{key}_k{k}": v for key, v in metrics.items()})
            all_metrics[f"mlp_k{k}"] = metrics
            log.info("MLP @%d: %s", k, metrics)

    with open(MODELS_DIR / "eval_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    log.info("Stage 4 concluído. Resultado salvo em %s/eval_metrics.json", MODELS_DIR)


if __name__ == "__main__":
    run()
