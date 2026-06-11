"""Testes do ModelService com artefatos falsos — sem modelo treinado em disco."""

import numpy as np
import pandas as pd
import pytest

from app.model_service import ModelService
from src.config.features import FEATURE_COLUMNS


class _IdentityTransformer:
    def transform(self, X):
        return np.asarray(X)


class _FakeModel:
    """Modelo falso: probabilidade = primeira feature (clipada em [0, 1])."""

    def predict(self, X):
        return np.clip(np.asarray(X)[:, [0]], 0.0, 1.0)


@pytest.fixture
def service() -> ModelService:
    svc = ModelService.__new__(ModelService)  # pula __init__ (não carrega artefatos)
    svc.features_imputer = _IdentityTransformer()
    svc.features_scaler = _IdentityTransformer()
    svc.model = _FakeModel()
    return svc


@pytest.fixture
def features() -> pd.DataFrame:
    data = pd.DataFrame(0.5, index=range(4), columns=FEATURE_COLUMNS)
    # probabilidades de "benigno": 0.9, 0.5, 0.4, 0.1
    data.iloc[:, 0] = [0.9, 0.5, 0.4, 0.1]
    return data


def test_predict_returns_expected_columns(service: ModelService, features: pd.DataFrame) -> None:
    result = service.predict(features)
    assert list(result.columns) == ["Prediction", "Label", "Probability (benign)"]
    assert len(result) == len(features)
    assert (result.index == features.index).all()


def test_threshold_decision(service: ModelService, features: pd.DataFrame) -> None:
    result = service.predict(features)
    # threshold 0.5: >= 0.5 -> benign (1); < 0.5 -> malignant (0)
    assert result["Prediction"].tolist() == [1, 1, 0, 0]
    assert result["Label"].tolist() == ["benign", "benign", "malignant", "malignant"]


def test_probability_passthrough(service: ModelService, features: pd.DataFrame) -> None:
    result = service.predict(features)
    np.testing.assert_allclose(result["Probability (benign)"], [0.9, 0.5, 0.4, 0.1], atol=1e-6)
