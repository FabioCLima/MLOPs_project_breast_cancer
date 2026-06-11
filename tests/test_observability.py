"""Testes do logging de predições e das métricas Prometheus."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.prediction_logger as plog
from app.api import create_app
from app.model_service import ModelService
from src.config.features import FEATURE_COLUMNS


class _IdentityTransformer:
    def transform(self, X):
        return np.asarray(X)


class _FakeModel:
    def predict(self, X):
        return np.clip(np.asarray(X)[:, [0]], 0.0, 1.0)


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setattr(plog, "PREDICTION_LOG_DIR", tmp_path)
    monkeypatch.setattr(plog, "PREDICTION_LOG_PATH", tmp_path / "predictions.jsonl")
    svc = ModelService.__new__(ModelService)
    svc.features_imputer = _IdentityTransformer()
    svc.features_scaler = _IdentityTransformer()
    svc.model = _FakeModel()
    svc.decision_threshold = 0.5
    svc.model_version = "test"
    app = create_app(model_service=svc)
    with TestClient(app) as client:
        yield client


def _valid_record(first_value: float = 0.9) -> dict:
    record = dict.fromkeys(FEATURE_COLUMNS, 0.5)
    record[FEATURE_COLUMNS[0]] = first_value
    return record


def test_predictions_are_logged_as_jsonl(client: TestClient, tmp_path) -> None:
    client.post("/predict", json={"records": [_valid_record(0.9), _valid_record(0.1)]})

    log_file = tmp_path / "predictions.jsonl"
    assert log_file.exists()
    lines = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert len(lines) == 2

    first = lines[0]
    assert first["model_version"] == "test"
    assert first["endpoint"] == "/predict"
    assert first["label"] == "benign"
    assert set(first["features"].keys()) == set(FEATURE_COLUMNS)
    assert "latency_ms_total" in first and "timestamp" in first


def test_metrics_endpoint_exposes_counters(client: TestClient) -> None:
    client.post("/predict", json={"records": [_valid_record(0.9)]})
    body = client.get("/metrics").text
    assert "predictions_total" in body
    assert "prediction_request_latency_seconds" in body
    assert "predicted_probability_benign" in body
