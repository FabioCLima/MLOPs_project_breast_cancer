"""Testes unitários dos estágios do pipeline (split, modelo, preparação)."""

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing.preprocess_data import split_data
from src.model_training.train_model import create_model, prepare_data


@pytest.fixture(scope="module")
def synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "target": (rng.random(n) < 0.6).astype(int),
        }
    )


def test_split_is_reproducible(synthetic_data: pd.DataFrame) -> None:
    train1, val1, test1 = split_data(synthetic_data)
    train2, val2, test2 = split_data(synthetic_data)
    pd.testing.assert_frame_equal(train1, train2)
    pd.testing.assert_frame_equal(val1, val2)
    pd.testing.assert_frame_equal(test1, test2)


def test_split_sizes(synthetic_data: pd.DataFrame) -> None:
    train, val, test = split_data(synthetic_data)
    n = len(synthetic_data)
    assert len(train) + len(val) + len(test) == n
    # test_size=0.2 do total; val_size=0.2 do restante (ver params.yaml)
    assert len(test) == pytest.approx(0.2 * n, abs=2)
    assert len(val) == pytest.approx(0.2 * 0.8 * n, abs=2)


def test_prepare_data_shapes(synthetic_data: pd.DataFrame) -> None:
    X, y = prepare_data(synthetic_data)
    assert X.shape == (len(synthetic_data), 2)
    assert y.shape == (len(synthetic_data),)
    assert set(y.unique()) <= {0, 1}
    assert "target" not in X.columns


def test_create_model_binary_output() -> None:
    params = {
        "hidden_layer_1_neurons": 8,
        "hidden_layer_2_neurons": 4,
        "dropout_rate": 0.1,
        "learning_rate": 0.01,
    }
    model = create_model(input_shape=30, params=params)
    assert model.output_shape == (None, 1)
    assert model.loss == "binary_crossentropy"


def test_model_trains_one_epoch() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 5)).astype("float32")
    y = (rng.random(64) < 0.5).astype("int32")
    params = {
        "hidden_layer_1_neurons": 8,
        "hidden_layer_2_neurons": 4,
        "dropout_rate": 0.1,
        "learning_rate": 0.01,
    }
    model = create_model(input_shape=5, params=params)
    history = model.fit(X, y, epochs=1, batch_size=32, verbose=0)
    assert "loss" in history.history
    proba = model.predict(X, verbose=0).ravel()
    assert ((proba >= 0) & (proba <= 1)).all()
