import json

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from src.config.logging_config import setup_logger
from src.config.params import load_params
from src.config.paths import (
    IMPUTER_PATH,
    MLFLOW_RUN_ID_PATH,
    MODEL_PATH,
    SCALER_PATH,
    TRAIN_PROCESSED_PATH,
    TRAINING_METRICS_PATH,
    VAL_PROCESSED_PATH,
)
from src.config.tracking import REGISTERED_MODEL_NAME, setup_mlflow
from src.data_validation.schemas import PROCESSED_SCHEMA, validate


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the feature-engineered train and validation data.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and validation datasets.
    """
    logger.info(f"Loading feature data from {TRAIN_PROCESSED_PATH} and {VAL_PROCESSED_PATH}")
    train_data = validate(pd.read_csv(TRAIN_PROCESSED_PATH), PROCESSED_SCHEMA, stage="train")
    val_data = validate(pd.read_csv(VAL_PROCESSED_PATH), PROCESSED_SCHEMA, stage="train")
    return train_data, val_data


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and binary target.

    Args:
        data (pd.DataFrame): Dataset with a `target` column.

    Returns:
        tuple containing:
            pd.DataFrame: Features
            pd.Series: Binary labels (0/1)
    """
    X = data.drop("target", axis=1)
    y = data["target"].astype(int)
    return X, y


def create_model(input_shape: int, params: dict[str, int | float]) -> tf.keras.Model:
    """Create a Keras binary classifier (MLP with sigmoid output).

    Args:
        input_shape (int): Number of input features.
        params (dict[str, int | float]): Model hyperparameters.

    Returns:
        tf.keras.Model: Compiled Keras model.
    """
    model = Sequential(
        [
            Input(shape=(input_shape,)),
            Dense(params["hidden_layer_1_neurons"], activation="relu"),
            Dropout(params["dropout_rate"]),
            Dense(params["hidden_layer_2_neurons"], activation="relu"),
            Dropout(params["dropout_rate"]),
            Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = Adam(learning_rate=params["learning_rate"])

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    return model


def save_training_artifacts(model: tf.keras.Model) -> None:
    """Save the trained model to disk.

    Args:
        model (tf.keras.Model): Trained Keras model.
    """
    logger.info(f"Saving model to {MODEL_PATH}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)


def train_model(
    train_data: pd.DataFrame, val_data: pd.DataFrame, params: dict[str, int | float]
) -> None:
    """Train a Keras model with an explicit, leakage-free validation set.

    Args:
        train_data (pd.DataFrame): Training dataset.
        val_data (pd.DataFrame): Validation dataset (held out before any transformer fit).
        params (dict[str, int | float]): Model hyperparameters.
    """
    tf.keras.utils.set_random_seed(int(params["random_seed"]))

    # Prepare the data
    X_train, y_train = prepare_data(train_data)
    X_val, y_val = prepare_data(val_data)

    # Create the model
    model = create_model(input_shape=X_train.shape[1], params=params)

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    setup_mlflow()
    with mlflow.start_run(run_name="train-keras-mlp") as run:
        mlflow.log_params(params)

        # Train the model with the explicit validation set
        logger.info("Training model...")
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=[early_stopping],
        )

        save_training_artifacts(model)

        # Curvas completas por época (UI do MLflow plota a evolução)
        for metric, values in history.history.items():
            for epoch, value in enumerate(values):
                mlflow.log_metric(metric, float(value), step=epoch)

        # Report metrics from the best epoch — the one whose weights were restored —
        # so the saved model and the reported metrics refer to the same state.
        best_epoch = int(np.argmin(history.history["val_loss"]))
        metrics = {metric: float(values[best_epoch]) for metric, values in history.history.items()}
        metrics["best_epoch"] = best_epoch
        mlflow.log_metrics({f"best_{k}": v for k, v in metrics.items()})

        # Modelo + artefatos de preprocessing na mesma run (lineage completo)
        mlflow.tensorflow.log_model(
            model, name="model", registered_model_name=REGISTERED_MODEL_NAME
        )
        mlflow.log_artifact(str(IMPUTER_PATH), artifact_path="preprocessing")
        mlflow.log_artifact(str(SCALER_PATH), artifact_path="preprocessing")

        # Persiste o run_id para o estágio de avaliação anexar métricas à mesma run
        MLFLOW_RUN_ID_PATH.parent.mkdir(parents=True, exist_ok=True)
        MLFLOW_RUN_ID_PATH.write_text(run.info.run_id)

    TRAINING_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Best epoch: {best_epoch} | val_loss: {metrics['val_loss']:.4f}")


def main() -> None:
    """Main function to orchestrate the model training process."""
    setup_logger()
    train_data, val_data = load_data()
    params = load_params("train")
    train_model(train_data, val_data, params)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()
