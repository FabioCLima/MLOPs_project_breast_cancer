"""Configuração centralizada do MLflow (tracking + model registry).

Backend local em SQLite (o model registry exige um store com banco de dados;
o file-store puro não suporta aliases/versões). Artefatos ficam em mlruns/.

Para abrir a UI:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import mlflow

from src.config.paths import PROJECT_ROOT

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"

EXPERIMENT_NAME = "breast-cancer-classifier"
REGISTERED_MODEL_NAME = "breast-cancer-classifier"

# Alias que o serving (Fase 3) usa para carregar o modelo de produção
PRODUCTION_ALIAS = "production"


def setup_mlflow() -> None:
    """Aponta o MLflow para o backend local e seleciona o experimento."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
