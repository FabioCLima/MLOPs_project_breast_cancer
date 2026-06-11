import joblib
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler

from src.config.logging_config import setup_logger
from src.config.paths import (
    SCALER_PATH,
    TEST_PREPROCESSED_PATH,
    TEST_PROCESSED_PATH,
    TRAIN_PREPROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
    VAL_PREPROCESSED_PATH,
    VAL_PROCESSED_PATH,
)
from src.data_validation.schemas import PREPROCESSED_SCHEMA, validate


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train, validation and test datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test datasets
    """
    logger.info(f"Loading preprocessed data from {TRAIN_PREPROCESSED_PATH.parent}")
    train = validate(pd.read_csv(TRAIN_PREPROCESSED_PATH), PREPROCESSED_SCHEMA, stage="features")
    val = validate(pd.read_csv(VAL_PREPROCESSED_PATH), PREPROCESSED_SCHEMA, stage="features")
    test = validate(pd.read_csv(TEST_PREPROCESSED_PATH), PREPROCESSED_SCHEMA, stage="features")
    return train, val, test


def engineer_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale features, fitting the scaler on the train split only.

    Args:
        train (pd.DataFrame): Training dataset
        val (pd.DataFrame): Validation dataset
        test (pd.DataFrame): Test dataset

    Returns:
        tuple containing:
            pd.DataFrame: Engineered training features
            pd.DataFrame: Engineered validation features
            pd.DataFrame: Engineered test features
            StandardScaler: Fitted scaler
    """
    logger.info("Engineering features...")
    feature_columns = [col for col in train.columns if col != "target"]

    scaler = StandardScaler()

    train_processed = train.copy()
    val_processed = val.copy()
    test_processed = test.copy()

    train_processed[feature_columns] = scaler.fit_transform(train_processed[feature_columns])
    val_processed[feature_columns] = scaler.transform(val_processed[feature_columns])
    test_processed[feature_columns] = scaler.transform(test_processed[feature_columns])

    return train_processed, val_processed, test_processed, scaler


def save_artifacts(
    train_processed: pd.DataFrame,
    val_processed: pd.DataFrame,
    test_processed: pd.DataFrame,
    scaler: StandardScaler,
) -> None:
    """Save engineered features and scaler.

    Args:
        train_processed (pd.DataFrame): Engineered training data
        val_processed (pd.DataFrame): Engineered validation data
        test_processed (pd.DataFrame): Engineered test data
        scaler (StandardScaler): Fitted scaler
    """
    # Save processed data
    logger.info(f"Saving engineered features to {TRAIN_PROCESSED_PATH.parent}")
    TRAIN_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_processed.to_csv(TRAIN_PROCESSED_PATH, index=False)
    val_processed.to_csv(VAL_PROCESSED_PATH, index=False)
    test_processed.to_csv(TEST_PROCESSED_PATH, index=False)

    # Save scaler
    logger.info(f"Saving scaler to {SCALER_PATH}")
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)


def main() -> None:
    """Main function to orchestrate feature engineering pipeline."""
    setup_logger()
    train, val, test = load_preprocessed_data()
    train_processed, val_processed, test_processed, scaler = engineer_features(train, val, test)
    save_artifacts(train_processed, val_processed, test_processed, scaler)
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
