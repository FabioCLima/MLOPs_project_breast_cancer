"""ModelService: encapsula artefatos de preprocessing + modelo para inferência.

Módulo sem efeitos colaterais de import — nada é carregado até ModelService()
ser instanciado. Isso permite testar a lógica de predição com artefatos falsos.
"""

import joblib
import pandas as pd
from loguru import logger
from tensorflow.keras.models import load_model

from src.config.features import TARGET_LABELS
from src.config.paths import IMPUTER_PATH, MODEL_PATH, SCALER_PATH

DECISION_THRESHOLD = 0.5


class ModelService:
    def __init__(self) -> None:
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all artifacts from the local project folder."""
        logger.info("Loading artifacts from local project folder")

        self.features_imputer = joblib.load(IMPUTER_PATH)
        self.features_scaler = joblib.load(SCALER_PATH)
        self.model = load_model(MODEL_PATH)

        logger.info("Successfully loaded all artifacts")

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
        y_pred = (y_pred_proba >= DECISION_THRESHOLD).astype(int)
        labels = pd.Series(y_pred, index=features.index).map(TARGET_LABELS)

        return pd.DataFrame(
            {
                "Prediction": y_pred,
                "Label": labels,
                "Probability (benign)": y_pred_proba.round(4),
            },
            index=features.index,
        )
