import json

import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from src.config.logging_config import setup_logger
from src.config.paths import EVALUATION_METRICS_PATH, MODEL_PATH, TEST_PROCESSED_PATH

DECISION_THRESHOLD = 0.5


def load_model() -> tf.keras.Model:
    """Load the trained Keras model from disk.

    Returns:
        tf.keras.Model: Loaded Keras model.
    """
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the test dataset from disk.

    Returns:
        tuple containing:
            pd.DataFrame: Test features
            pd.Series: Test labels
    """
    logger.info(f"Loading test data from {TEST_PROCESSED_PATH}")
    data = pd.read_csv(TEST_PROCESSED_PATH)
    X = data.drop("target", axis=1)
    y = data["target"].astype(int)
    return X, y


def evaluate_model(model: tf.keras.Model, X: pd.DataFrame, y_true: pd.Series) -> None:
    """Evaluate the model and generate performance metrics.

    Args:
        model (tf.keras.Model): Trained Keras model.
        X (pd.DataFrame): Test features.
        y_true (pd.Series): True labels.
    """
    # Generate model predictions (sigmoid probability of class 1 = benign)
    y_pred_proba = model.predict(X).ravel()
    y_pred = (y_pred_proba >= DECISION_THRESHOLD).astype(int)

    # Calculate evaluation metrics
    report = classification_report(
        y_true, y_pred, target_names=["malignant", "benign"], output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    evaluation = {
        "decision_threshold": DECISION_THRESHOLD,
        "classification_report": report,
        "confusion_matrix": cm,
    }

    # Log metrics
    logger.info(
        "Classification Report:\n"
        f"{classification_report(y_true, y_pred, target_names=['malignant', 'benign'])}"
    )
    EVALUATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)


def main() -> None:
    """Main function to orchestrate the model evaluation process."""
    setup_logger()
    model = load_model()
    X, y = load_test_data()
    evaluate_model(model, X, y)
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()
