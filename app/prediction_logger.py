"""Logging estruturado de predições — a matéria-prima do monitoramento.

Cada predição vira uma linha JSON (JSONL) com: timestamp, versão do modelo,
threshold, latência, as 30 features de entrada e a saída. É deste arquivo que
o relatório de drift (item 23) e a avaliação em produção (item 25) se alimentam.

O destino é configurável por env var (PREDICTION_LOG_DIR) para que, em
container/K8s, aponte para um volume — nunca para o filesystem efêmero.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from src.config.paths import LOGS_DIR

PREDICTION_LOG_DIR = Path(os.getenv("PREDICTION_LOG_DIR", str(LOGS_DIR / "predictions")))
PREDICTION_LOG_PATH = PREDICTION_LOG_DIR / "predictions.jsonl"


def log_predictions(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    model_version: str,
    decision_threshold: float,
    latency_ms: float,
    endpoint: str,
) -> None:
    """Registra cada predição como uma linha JSON.

    Args:
        features: entradas (30 colunas, nomes originais).
        predictions: saída do ModelService (Prediction, Label, Probability).
        model_version: versão servida (auditoria por versão).
        decision_threshold: threshold aplicado.
        latency_ms: latência total do request (rateada por linha no batch).
        endpoint: rota que originou a predição.
    """
    PREDICTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(tz=UTC).isoformat()

    with open(PREDICTION_LOG_PATH, "a") as f:
        for (_, feat_row), (_, pred_row) in zip(
            features.iterrows(), predictions.iterrows(), strict=True
        ):
            record = {
                "timestamp": now,
                "endpoint": endpoint,
                "model_version": model_version,
                "decision_threshold": decision_threshold,
                "latency_ms_total": round(latency_ms, 2),
                "features": {k: float(v) for k, v in feat_row.items()},
                "prediction": int(pred_row["Prediction"]),
                "label": str(pred_row["Label"]),
                "probability_benign": float(pred_row["Probability (benign)"]),
            }
            f.write(json.dumps(record) + "\n")

    logger.info(
        f"{endpoint}: {len(predictions)} predição(ões) | "
        f"malignas: {(predictions['Prediction'] == 0).sum()} | "
        f"{latency_ms:.0f} ms | modelo {model_version}"
    )
