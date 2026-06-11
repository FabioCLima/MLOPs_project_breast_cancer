"""Provas automatizadas de que os transformadores não enxergam validação/teste.

Data leakage de preprocessing acontece quando estatísticas (média do imputer,
média/desvio do scaler) são calculadas usando linhas que depois serão usadas
para validar ou testar o modelo. Estes testes falham se isso voltar a ocorrer.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing.preprocess_data import preprocess_data, split_data
from src.feature_engineering.engineer_features import engineer_features


@pytest.fixture(scope="module")
def synthetic_data() -> pd.DataFrame:
    """Dataset sintético com NaNs e duas classes desbalanceadas (70/30)."""
    rng = np.random.default_rng(0)
    n = 500
    data = pd.DataFrame(
        {
            "feat_a": rng.normal(10, 2, n),
            "feat_b": rng.normal(-5, 1, n),
            "target": (rng.random(n) < 0.7).astype(int),
        }
    )
    data.loc[rng.random(n) < 0.05, "feat_a"] = np.nan
    return data


@pytest.fixture(scope="module")
def splits(synthetic_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return split_data(synthetic_data)


def test_splits_are_disjoint(splits) -> None:
    train, val, test = splits
    assert set(train.index).isdisjoint(val.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(val.index).isdisjoint(test.index)


def test_splits_are_stratified(splits, synthetic_data) -> None:
    overall = synthetic_data["target"].mean()
    for split in splits:
        assert split["target"].mean() == pytest.approx(overall, abs=0.03)


def test_imputer_fitted_on_train_only(splits) -> None:
    train, val, test = splits
    _, _, _, imputer = preprocess_data(train, val, test)
    train_means = train.drop("target", axis=1).mean().to_numpy()
    full = pd.concat([train, val, test])
    full_means = full.drop("target", axis=1).mean().to_numpy()

    np.testing.assert_allclose(imputer.statistics_, train_means, rtol=1e-9)
    assert not np.allclose(imputer.statistics_, full_means)


def test_scaler_fitted_on_train_only(splits) -> None:
    train, val, test = splits
    train_p, val_p, test_p, _ = preprocess_data(train, val, test)
    _, _, _, scaler = engineer_features(train_p, val_p, test_p)
    train_means = train_p.drop("target", axis=1).mean().to_numpy()
    full = pd.concat([train_p, val_p, test_p])
    full_means = full.drop("target", axis=1).mean().to_numpy()

    np.testing.assert_allclose(scaler.mean_, train_means, rtol=1e-9)
    assert not np.allclose(scaler.mean_, full_means)


def test_scaled_train_is_standardized_but_val_test_are_not_exactly(splits) -> None:
    """O treino escalado tem média ~0 por construção; val/teste só aproximadamente.

    Se val/teste tivessem média exatamente 0, o scaler teria sido ajustado neles —
    ou seja, leakage.
    """
    train, val, test = splits
    train_p, val_p, test_p, _ = preprocess_data(train, val, test)
    train_s, val_s, test_s, _ = engineer_features(train_p, val_p, test_p)

    features = [c for c in train_s.columns if c != "target"]
    np.testing.assert_allclose(train_s[features].mean(), 0.0, atol=1e-9)
    assert (val_s[features].mean().abs() > 1e-9).all()
    assert (test_s[features].mean().abs() > 1e-9).all()
