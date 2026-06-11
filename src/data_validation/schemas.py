"""Schemas Pandera — o contrato de dados entre os estágios do pipeline.

Três contratos, um por fronteira:

- RAW:           30 features não-negativas (NaN permitido), target binário.
- PREPROCESSED:  como RAW, mas sem NaN (pós-imputação).
- PROCESSED:     features escaladas (podem ser negativas), sem NaN, target binário.

Validar com `lazy=True` acumula todas as violações num único relatório,
em vez de parar na primeira.
"""

import pandas as pd
import pandera.pandas as pa

from src.config.features import FEATURE_COLUMNS, TARGET_COLUMN

_TARGET_COLUMN_SCHEMA = pa.Column(int, checks=pa.Check.isin([0, 1]))


def _feature_schema(nullable: bool, non_negative: bool) -> dict[str, pa.Column]:
    checks = [pa.Check.ge(0)] if non_negative else []
    return {
        name: pa.Column(float, checks=checks, nullable=nullable, coerce=True)
        for name in FEATURE_COLUMNS
    }


RAW_SCHEMA = pa.DataFrameSchema(
    columns={
        **_feature_schema(nullable=True, non_negative=True),
        TARGET_COLUMN: _TARGET_COLUMN_SCHEMA,
    },
    strict=True,
)

PREPROCESSED_SCHEMA = pa.DataFrameSchema(
    columns={
        **_feature_schema(nullable=False, non_negative=True),
        TARGET_COLUMN: _TARGET_COLUMN_SCHEMA,
    },
    strict=True,
)

PROCESSED_SCHEMA = pa.DataFrameSchema(
    columns={
        **_feature_schema(nullable=False, non_negative=False),
        TARGET_COLUMN: _TARGET_COLUMN_SCHEMA,
    },
    strict=True,
)


def validate(data: pd.DataFrame, schema: pa.DataFrameSchema, stage: str) -> pd.DataFrame:
    """Valida um DataFrame contra um schema, com erro acionável.

    Args:
        data: DataFrame a validar.
        schema: Um dos schemas deste módulo.
        stage: Nome do estágio (aparece na mensagem de erro).

    Returns:
        O DataFrame validado (com dtypes coagidos pelo schema).

    Raises:
        pa.errors.SchemaErrors: relatório completo das violações.
    """
    try:
        return schema.validate(data, lazy=True)
    except pa.errors.SchemaErrors as err:
        raise pa.errors.SchemaError(
            schema, data, f"Contrato de dados violado no estágio '{stage}':\n{err.failure_cases}"
        ) from err
