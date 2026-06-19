"""Stage 4 do pipeline DVC — avaliação final no conjunto de teste."""

import json
import logging
import pickle
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from configs.settings import settings
from src.models.baseline import PopularityBaseline
from src.models.mlp import RecommenderMLP
from src.training.trainer import MLPTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
MODELS_DIR = Path("models")
PARAMS_FILE = Path("params.yaml")


def load_params() -> dict:
    """Carrega todos os hiperparâmetros do params.yaml.

    Returns:
        Dicionário com os parâmetros de treino, modelo e avaliação.
    """
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)


def rebuild_mlp(params: dict, train: pd.DataFrame) -> RecommenderMLP:
    """Reconstrói a arquitetura do MLP usada no treino original.

    Necessário porque salvamos apenas os pesos (state_dict), não a
    arquitetura completa do modelo.

    Args:
        params: Hiperparâmetros do modelo (params.yaml).
        train: DataFrame de treino, usado para inferir n_users e
            n_products a partir dos IDs codificados.

    Returns:
        Instância de RecommenderMLP com a arquitetura correta,
        ainda sem os pesos carregados.
    """
    n_users = int(train["user_id_enc"].max()) + 1
    n_products = int(train["product_id_enc"].max()) + 1
    return RecommenderMLP(
        n_users=n_users,
        n_products=n_products,
        embedding_dim=params["model"]["embedding_dim"],
        hidden_layers=params["model"]["hidden_layers"],
        dropout=params["model"]["dropout"],
    )


def evaluate_at_k_values(
    evaluator,
    test: pd.DataFrame,
    k_values: list[int],
    run_name: str,
    metric_prefix: str,
) -> dict:
    """Avalia um modelo para múltiplos valores de K e loga no MLflow.

    Args:
        evaluator: Modelo com método evaluate(test, k) -> dict.
        test: DataFrame de teste.
        k_values: Lista de valores de K a avaliar.
        run_name: Nome do MLflow run.
        metric_prefix: Prefixo usado nas chaves do dicionário retornado.

    Returns:
        Dicionário {f"{metric_prefix}_k{k}": métricas} para cada K.
    """
    results: dict = {}
    with mlflow.start_run(run_name=run_name):
        for k in k_values:
            metrics = evaluator.evaluate(test, k=k)
            mlflow.log_metrics({f"{key}_k{k}": v for key, v in metrics.items()})
            results[f"{metric_prefix}_k{k}"] = metrics
            log.info("%s @%d: %s", metric_prefix, k, metrics)
    return results


def setup_mlflow() -> None:
    """Configura o tracking URI e o experimento do MLflow."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("tech-challenge-recommendation")


def load_models(
    params: dict,
    train: pd.DataFrame,
) -> tuple[PopularityBaseline, MLPTrainer]:
    """Carrega o baseline e reconstrói o MLP a partir dos checkpoints.

    Args:
        params: Hiperparâmetros do modelo (params.yaml).
        train: DataFrame de treino, usado para inferir as dimensões
            de embedding do MLP.

    Returns:
        Tupla (baseline, trainer) prontos para avaliação.
    """
    with open(MODELS_DIR / "baseline.pkl", "rb") as f:
        baseline: PopularityBaseline = pickle.load(f)

    model = rebuild_mlp(params, train)
    trainer = MLPTrainer.load(MODELS_DIR / "mlp_best.pt", model=model)
    return baseline, trainer


def save_eval_metrics(baseline_results: dict, mlp_results: dict) -> None:
    """Combina e persiste as métricas finais de avaliação.

    Args:
        baseline_results: Métricas do PopularityBaseline por K.
        mlp_results: Métricas do RecommenderMLP por K.
    """
    all_metrics = {**baseline_results, **mlp_results}
    with open(MODELS_DIR / "eval_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    log.info("Resultado salvo em %s/eval_metrics.json", MODELS_DIR)


def run() -> None:
    """Avalia baseline e MLP no teste e registra métricas no MLflow."""
    test = pd.read_parquet(FEATURES_DIR / "test.parquet")
    train = pd.read_parquet(FEATURES_DIR / "train.parquet")
    params = load_params()
    k_values = params["evaluate"]["k_values"]

    baseline, trainer = load_models(params, train)
    setup_mlflow()

    log.info("Avaliando PopularityBaseline no conjunto de teste...")
    baseline_results = evaluate_at_k_values(
        baseline, test, k_values, "eval-baseline", "baseline"
    )

    log.info("Avaliando RecommenderMLP no conjunto de teste...")
    mlp_results = evaluate_at_k_values(trainer, test, k_values, "eval-mlp", "mlp")

    save_eval_metrics(baseline_results, mlp_results)
    log.info("Stage 4 concluído.")


if __name__ == "__main__":
    run()
