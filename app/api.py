"""API de predição (FastAPI).

Application factory + lifespan: o modelo é carregado na inicialização do
servidor, nunca no import — importar este módulo não exige modelo treinado.

Endpoints:
    GET  /health         — liveness/readiness (app de pé; modelo carregado?)
    GET  /metadata       — versão do modelo, threshold, features, métricas
    POST /predict        — JSON (lote de registros validados pelo Pydantic)
    POST /predict/batch  — upload de CSV
"""

import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, UploadFile
from loguru import logger

from app.model_service import ModelService
from app.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
)
from src.config.features import FEATURE_COLUMNS
from src.config.logging_config import setup_logger
from src.config.paths import EVALUATION_METRICS_PATH
from src.config.tracking import REGISTERED_MODEL_NAME


def create_app(model_service: ModelService | None = None) -> FastAPI:
    """Cria a aplicação. `model_service` injetável para testes.

    Args:
        model_service: serviço pré-construído (testes); se None, carrega no startup.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logger()
        app.state.model_service = model_service or ModelService()
        logger.info("API pronta para receber requisições")
        yield

    app = FastAPI(
        title="Breast Cancer Classifier API",
        description="Predição de malignidade (didático — não usar para decisão clínica)",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        service = getattr(request.app.state, "model_service", None)
        return HealthResponse(status="ok", model_loaded=service is not None)

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata(request: Request) -> MetadataResponse:
        service: ModelService = request.app.state.model_service
        try:
            with open(EVALUATION_METRICS_PATH) as f:
                report = json.load(f)["classification_report"]
            test_metrics = {
                "accuracy": report["accuracy"],
                "malignant_recall": report["malignant"]["recall"],
                "malignant_precision": report["malignant"]["precision"],
            }
        except FileNotFoundError:
            test_metrics = {}
        return MetadataResponse(
            model_name=REGISTERED_MODEL_NAME,
            model_version=service.model_version,
            decision_threshold=service.decision_threshold,
            feature_names=FEATURE_COLUMNS,
            test_metrics=test_metrics,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: Request, payload: PredictionRequest) -> PredictionResponse:
        service: ModelService = request.app.state.model_service
        # by_alias=True restaura os nomes originais ("mean radius"), que são
        # os nomes com que imputer/scaler foram ajustados
        features = pd.DataFrame(
            [record.model_dump(by_alias=True) for record in payload.records]  # type: ignore[attr-defined]
        )[FEATURE_COLUMNS]
        predictions = service.predict(features)
        return _to_response(service, predictions)

    @app.post("/predict/batch", response_model=PredictionResponse)
    async def predict_batch(request: Request, file: UploadFile) -> PredictionResponse:
        service: ModelService = request.app.state.model_service
        if not file.filename or not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Envie um arquivo .csv")

        content = (await file.read()).decode("utf-8")
        try:
            features = pd.read_csv(io.StringIO(content))
        except Exception as err:
            raise HTTPException(status_code=400, detail=f"CSV inválido: {err}") from err

        missing = [col for col in FEATURE_COLUMNS if col not in features.columns]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Colunas obrigatórias ausentes: {', '.join(missing)}"
            )

        predictions = service.predict(features[FEATURE_COLUMNS])
        return _to_response(service, predictions)

    return app


def _to_response(service: ModelService, predictions: pd.DataFrame) -> PredictionResponse:
    results = [
        PredictionResult(
            prediction=int(row["Prediction"]),
            label=str(row["Label"]),
            probability_benign=float(row["Probability (benign)"]),
        )
        for _, row in predictions.iterrows()
    ]
    return PredictionResponse(
        model_version=service.model_version,
        decision_threshold=service.decision_threshold,
        results=results,
    )


def main() -> None:
    """Run the development server."""
    import uvicorn  # noqa: PLC0415 — dependência só do entrypoint dev

    uvicorn.run(create_app(), host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
