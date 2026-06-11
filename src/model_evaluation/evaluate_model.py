import json

import mlflow
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix

from src.config.logging_config import setup_logger
from src.config.params import load_params
from src.config.paths import (
    EVALUATION_METRICS_PATH,
    MLFLOW_RUN_ID_PATH,
    MODEL_PATH,
    TEST_PROCESSED_PATH,
)
from src.config.tracking import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, setup_mlflow
from src.data_validation.schemas import PROCESSED_SCHEMA, validate

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
    data = validate(pd.read_csv(TEST_PROCESSED_PATH), PROCESSED_SCHEMA, stage="evaluate")
    X = data.drop("target", axis=1)
    y = data["target"].astype(int)
    return X, y


def evaluate_model(model: tf.keras.Model, X: pd.DataFrame, y_true: pd.Series) -> dict:
    """Evaluate the model and generate performance metrics.

    Args:
        model (tf.keras.Model): Trained Keras model.
        X (pd.DataFrame): Test features.
        y_true (pd.Series): True labels.

    Returns:
        dict: evaluation payload (threshold, classification report, confusion matrix).
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
    return evaluation


def track_and_promote(evaluation: dict) -> None:
    """Anexa métricas de teste à run de treino e aplica o gate de promoção.

    O modelo recém-registrado só recebe o alias de produção se o recall da
    classe maligna no teste atingir o mínimo definido em params.yaml.
    """
    setup_mlflow()
    report = evaluation["classification_report"]
    test_metrics = {
        "test_accuracy": report["accuracy"],
        "test_malignant_recall": report["malignant"]["recall"],
        "test_malignant_precision": report["malignant"]["precision"],
        "test_benign_recall": report["benign"]["recall"],
        "test_f1_macro": report["macro avg"]["f1-score"],
    }

    run_id = MLFLOW_RUN_ID_PATH.read_text().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(test_metrics)

    # Gate de promoção: alias "production" só com recall maligno suficiente
    min_recall = load_params("evaluate")["min_malignant_recall"]
    client = mlflow.MlflowClient()
    latest_version = max(
        int(v.version) for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    )
    if test_metrics["test_malignant_recall"] >= min_recall:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, str(latest_version)
        )
        logger.success(
            f"Model v{latest_version} promoted to '{PRODUCTION_ALIAS}' "
            f"(malignant recall {test_metrics['test_malignant_recall']:.3f} >= {min_recall})"
        )
    else:
        logger.warning(
            f"Model v{latest_version} NOT promoted: malignant recall "
            f"{test_metrics['test_malignant_recall']:.3f} < {min_recall}"
        )


def main() -> None:
    """Main function to orchestrate the model evaluation process."""
    setup_logger()
    model = load_model()
    X, y = load_test_data()
    evaluation = evaluate_model(model, X, y)
    track_and_promote(evaluation)
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()
