"""Testes do contrato de dados (Pandera) entre os estágios do pipeline."""

import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN
from src.data_validation.schemas import PREPROCESSED_SCHEMA, RAW_SCHEMA, validate


@pytest.fixture
def valid_raw_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.uniform(0, 100, size=(20, 30)), columns=FEATURE_COLUMNS)
    data.iloc[0, 0] = np.nan  # NaN permitido no raw
    data[TARGET_COLUMN] = rng.integers(0, 2, size=20)
    return data


def test_valid_raw_data_passes(valid_raw_data: pd.DataFrame) -> None:
    validated = validate(valid_raw_data, RAW_SCHEMA, stage="test")
    assert validated.shape == valid_raw_data.shape


def test_missing_column_fails(valid_raw_data: pd.DataFrame) -> None:
    broken = valid_raw_data.drop(columns=["mean radius"])
    with pytest.raises(pa.errors.SchemaError, match="test"):
        validate(broken, RAW_SCHEMA, stage="test")


def test_extra_column_fails(valid_raw_data: pd.DataFrame) -> None:
    broken = valid_raw_data.assign(injected="boom")
    with pytest.raises(pa.errors.SchemaError):
        validate(broken, RAW_SCHEMA, stage="test")


def test_negative_feature_fails(valid_raw_data: pd.DataFrame) -> None:
    broken = valid_raw_data.copy()
    broken.loc[0, "mean area"] = -1.0
    with pytest.raises(pa.errors.SchemaError):
        validate(broken, RAW_SCHEMA, stage="test")


def test_non_binary_target_fails(valid_raw_data: pd.DataFrame) -> None:
    broken = valid_raw_data.copy()
    broken.loc[0, TARGET_COLUMN] = 2
    with pytest.raises(pa.errors.SchemaError):
        validate(broken, RAW_SCHEMA, stage="test")


def test_nan_rejected_after_imputation(valid_raw_data: pd.DataFrame) -> None:
    # O mesmo dado com NaN passa no RAW mas falha no PREPROCESSED
    with pytest.raises(pa.errors.SchemaError):
        validate(valid_raw_data, PREPROCESSED_SCHEMA, stage="test")
