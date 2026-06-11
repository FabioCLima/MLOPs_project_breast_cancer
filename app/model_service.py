"""ModelService: encapsula artefatos de preprocessing + modelo para inferência.

Módulo sem efeitos colaterais de import — nada é carregado até ModelService()
ser instanciado. Isso permite testar a lógica de predição com artefatos falsos.

Origem do modelo, em ordem de preferência:
1. MLflow Model Registry, alias "production" (models:/<nome>@production)
2. Fallback: arquivo local models/model.keras (com warning)

O threshold de decisão vem de metrics/evaluation.json — o mesmo ponto da
curva precision-recall que a avaliação aprovou (evita skew avaliação/serving).
"""

import json

import joblib
import pandas as pd
from loguru import logger
from tensorflow.keras.models import load_model

from src.config.features import TARGET_LABELS
from src.config.paths import EVALUATION_METRICS_PATH, IMPUTER_PATH, MODEL_PATH, SCALER_PATH
from src.config.tracking import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, setup_mlflow

DEFAULT_THRESHOLD = 0.5


def _load_threshold() -> float:
    """Threshold ajustado pela avaliação; fallback 0.5 com warning."""
    try:
        with open(EVALUATION_METRICS_PATH) as f:
            return float(json.load(f)["decision_threshold"])
    except (FileNotFoundError, KeyError, ValueError):
        logger.warning(
            f"Threshold não encontrado em {EVALUATION_METRICS_PATH}; usando {DEFAULT_THRESHOLD}"
        )
        return DEFAULT_THRESHOLD


class ModelService:
    def __init__(self) -> None:
        self.decision_threshold = _load_threshold()
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Carrega preprocessadores locais e o modelo (registry -> fallback local)."""
        logger.info("Loading preprocessing artifacts")
        self.features_imputer = joblib.load(IMPUTER_PATH)
        self.features_scaler = joblib.load(SCALER_PATH)

        try:
            import mlflow  # noqa: PLC0415 — import tardio: fallback local não exige mlflow

            setup_mlflow()
            model_uri = f"models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}"
            self.model = mlflow.tensorflow.load_model(model_uri)
            version = mlflow.MlflowClient().get_model_version_by_alias(
                REGISTERED_MODEL_NAME, PRODUCTION_ALIAS
            )
            self.model_version = f"v{version.version}"
            logger.success(f"Model loaded from registry: {model_uri} ({self.model_version})")
        except Exception as err:
            logger.warning(f"Registry indisponível ({err}); fallback para {MODEL_PATH}")
            self.model = load_model(MODEL_PATH)
            self.model_version = "local-file"

        logger.info(
            f"ModelService pronto (modelo {self.model_version}, "
            f"threshold {self.decision_threshold:.2f})"
        )

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using the full pipeline.

        Args:
            features: DataFrame containing the input features

        Returns:
            DataFrame with prediction, label and probability per row
        """
        # Apply transformations in sequence
        X_imputed = self.features_imputer.transform(features)
        X_scaled = self.features_scaler.transform(X_imputed)

        # Sigmoid probability of class 1 (benign); explicit threshold decision
        y_pred_proba = self.model.predict(X_scaled).ravel()
        y_pred = (y_pred_proba >= self.decision_threshold).astype(int)
        labels = pd.Series(y_pred, index=features.index).map(TARGET_LABELS)

        return pd.DataFrame(
            {
                "Prediction": y_pred,
                "Label": labels,
                "Probability (benign)": y_pred_proba.round(4),
            },
            index=features.index,
        )
