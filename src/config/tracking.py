"""Configuração centralizada do MLflow (tracking + model registry).

Backend padrão: SQLite local (o model registry exige um store com banco de
dados; o file-store puro não suporta aliases/versões). Artefatos em mlruns/.

A env var MLFLOW_TRACKING_URI sobrepõe o padrão — é assim que o mesmo código
aponta para o MLflow server do docker-compose (http://localhost:5000) ou para
um servidor remoto na cloud, sem mudança de código.

Para abrir a UI local:
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import os

import mlflow

from src.config.paths import PROJECT_ROOT

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")

EXPERIMENT_NAME = "breast-cancer-classifier"
REGISTERED_MODEL_NAME = "breast-cancer-classifier"

# Alias que o serving (Fase 3) usa para carregar o modelo de produção
PRODUCTION_ALIAS = "production"


def setup_mlflow() -> None:
    """Aponta o MLflow para o backend local e seleciona o experimento."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
