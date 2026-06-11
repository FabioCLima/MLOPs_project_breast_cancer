"""Contrato Pydantic v2 da API de predição.

As 30 features são geradas dinamicamente a partir do contrato canônico
(src.config.features.FEATURE_COLUMNS) — o mesmo usado pelo pipeline e pelo
Pandera. Uma única fonte de verdade elimina training-serving skew de schema.

Os nomes originais têm espaços ("mean radius"); o campo Python usa underscore
(mean_radius) e o alias preserva o nome original. A API aceita ambos.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from src.config.features import FEATURE_COLUMNS

_field_definitions: dict[str, Any] = {
    name.replace(" ", "_"): (
        float,
        Field(ge=0, alias=name, description=f"Feature '{name}' (não-negativa)"),
    )
    for name in FEATURE_COLUMNS
}

PatientFeatures = create_model(  # type: ignore[call-overload]
    "PatientFeatures",
    __config__=ConfigDict(populate_by_name=True, extra="forbid"),
    **_field_definitions,
)


class PredictionRequest(BaseModel):
    """Lote de pacientes para predição."""

    records: list[PatientFeatures] = Field(min_length=1, max_length=1000)  # type: ignore[valid-type]


class PredictionResult(BaseModel):
    """Resultado de predição para um paciente."""

    prediction: int = Field(description="0 = malignant, 1 = benign")
    label: str
    probability_benign: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    """Resposta da API: resultados + proveniência do modelo."""

    model_version: str
    decision_threshold: float
    results: list[PredictionResult]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    decision_threshold: float
    feature_names: list[str]
    test_metrics: dict[str, float]
