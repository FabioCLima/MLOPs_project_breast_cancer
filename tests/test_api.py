"""Testes de contrato da API FastAPI — com ModelService falso injetado.

Nenhum teste aqui exige modelo treinado, registry ou artefatos em disco:
o create_app(model_service=...) recebe um serviço fake. É o payoff do
application factory.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.model_service import ModelService
from src.config.features import FEATURE_COLUMNS


class _IdentityTransformer:
    def transform(self, X):
        return np.asarray(X)


class _FakeModel:
    def predict(self, X):
        return np.clip(np.asarray(X)[:, [0]], 0.0, 1.0)


@pytest.fixture(scope="module")
def client() -> TestClient:
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


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_metadata(client: TestClient) -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test"
    assert body["decision_threshold"] == 0.5
    assert body["feature_names"] == FEATURE_COLUMNS


def test_predict_valid_payload(client: TestClient) -> None:
    payload = {"records": [_valid_record(0.9), _valid_record(0.1)]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test"
    assert [r["label"] for r in body["results"]] == ["benign", "malignant"]
    assert body["results"][0]["probability_benign"] == pytest.approx(0.9)


def test_predict_missing_feature_returns_422(client: TestClient) -> None:
    record = _valid_record()
    record.pop("mean radius")
    response = client.post("/predict", json={"records": [record]})
    assert response.status_code == 422
    assert "mean radius" in response.text


def test_predict_wrong_type_returns_422(client: TestClient) -> None:
    record = _valid_record()
    record["mean texture"] = "not-a-number"
    response = client.post("/predict", json={"records": [record]})
    assert response.status_code == 422


def test_predict_negative_feature_returns_422(client: TestClient) -> None:
    record = _valid_record()
    record["mean area"] = -10.0
    response = client.post("/predict", json={"records": [record]})
    assert response.status_code == 422


def test_predict_extra_field_returns_422(client: TestClient) -> None:
    record = _valid_record() | {"injected_column": 1.0}
    response = client.post("/predict", json={"records": [record]})
    assert response.status_code == 422


def test_predict_empty_records_returns_422(client: TestClient) -> None:
    response = client.post("/predict", json={"records": []})
    assert response.status_code == 422


def test_batch_csv(client: TestClient) -> None:
    df = pd.DataFrame([_valid_record(0.9), _valid_record(0.2)])
    response = client.post(
        "/predict/batch", files={"file": ("data.csv", df.to_csv(index=False), "text/csv")}
    )
    assert response.status_code == 200
    assert [r["label"] for r in response.json()["results"]] == ["benign", "malignant"]


def test_batch_missing_column_returns_400(client: TestClient) -> None:
    df = pd.DataFrame([_valid_record()]).drop(columns=["mean radius"])
    response = client.post(
        "/predict/batch", files={"file": ("data.csv", df.to_csv(index=False), "text/csv")}
    )
    assert response.status_code == 400
    assert "mean radius" in response.json()["detail"]


def test_batch_non_csv_returns_400(client: TestClient) -> None:
    response = client.post("/predict/batch", files={"file": ("data.txt", "hello", "text/plain")})
    assert response.status_code == 400
