import logging
import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("src.feature_engineering.engineer_features")


def load_preprocessed_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train, validation and test datasets.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test datasets
    """
    data_dir = "data/preprocessed"
    logger.info(f"Loading preprocessed data from {data_dir}")
    train = pd.read_csv(os.path.join(data_dir, "train_preprocessed.csv"))
    val = pd.read_csv(os.path.join(data_dir, "val_preprocessed.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test_preprocessed.csv"))
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
    output_dir = "data/processed"
    logger.info(f"Saving engineered features to {output_dir}")

    train_processed.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    val_processed.to_csv(os.path.join(output_dir, "val_processed.csv"), index=False)
    test_processed.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)

    # Save scaler
    scaler_path = os.path.join("artifacts", "[features]_scaler.joblib")
    logger.info(f"Saving scaler to {scaler_path}")
    joblib.dump(scaler, scaler_path)


def main() -> None:
    """Main function to orchestrate feature engineering pipeline."""
    train, val, test = load_preprocessed_data()
    train_processed, val_processed, test_processed, scaler = engineer_features(train, val, test)
    save_artifacts(train_processed, val_processed, test_processed, scaler)
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
