import logging

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.config.params import load_params
from src.config.paths import (
    IMPUTER_PATH,
    RAW_DATA_PATH,
    TEST_PREPROCESSED_PATH,
    TRAIN_PREPROCESSED_PATH,
    VAL_PREPROCESSED_PATH,
)

logger = logging.getLogger("src.data_preprocessing.preprocess_data")


def load_data() -> pd.DataFrame:
    """Load the raw data from disk.

    Returns:
        pd.DataFrame: Raw input data
    """
    logger.info(f"Loading raw data from {RAW_DATA_PATH}")
    data = pd.read_csv(RAW_DATA_PATH)
    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation and test sets, stratified by target.

    The validation set is carved out here — before any transformer is fitted —
    so that imputer/scaler statistics never see validation or test rows.

    Args:
        data (pd.DataFrame): Input dataset

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test datasets
    """
    params = load_params("preprocess_data")
    logger.info("Splitting data into train, validation and test sets (stratified)...")
    train_val_data, test_data = train_test_split(
        data,
        test_size=params["test_size"],
        random_state=params["random_seed"],
        stratify=data["target"],
    )
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=params["val_size"],
        random_state=params["random_seed"],
        stratify=train_val_data["target"],
    )
    return train_data, val_data, test_data


def preprocess_data(
    train_data: pd.DataFrame, val_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SimpleImputer]:
    """Impute missing values, fitting the imputer on the train split only.

    Args:
        train_data (pd.DataFrame): Training dataset
        val_data (pd.DataFrame): Validation dataset
        test_data (pd.DataFrame): Test dataset

    Returns:
        Tuple containing:
            pd.DataFrame: Processed training data
            pd.DataFrame: Processed validation data
            pd.DataFrame: Processed test data
            SimpleImputer: Fitted imputer
    """
    logger.info("Preprocessing data...")

    imputer = SimpleImputer(strategy="mean")

    def impute(split: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        target = split["target"]
        features = split.drop("target", axis=1)
        values = imputer.fit_transform(features) if fit else imputer.transform(features)
        processed = pd.DataFrame(values, columns=features.columns)
        return processed.assign(target=target.tolist())

    train_processed = impute(train_data, fit=True)
    val_processed = impute(val_data)
    test_processed = impute(test_data)

    return train_processed, val_processed, test_processed, imputer


def save_artifacts(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    imputer: SimpleImputer,
) -> None:
    """Save processed data and preprocessing artifacts.

    Args:
        train_data (pd.DataFrame): Processed training data
        val_data (pd.DataFrame): Processed validation data
        test_data (pd.DataFrame): Processed test data
        imputer (SimpleImputer): Fitted imputer
    """
    # Save processed data
    logger.info(f"Saving processed data to {TRAIN_PREPROCESSED_PATH.parent}")
    TRAIN_PREPROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(TRAIN_PREPROCESSED_PATH, index=False)
    val_data.to_csv(VAL_PREPROCESSED_PATH, index=False)
    test_data.to_csv(TEST_PREPROCESSED_PATH, index=False)

    # Save imputer
    logger.info(f"Saving imputer to {IMPUTER_PATH}")
    IMPUTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(imputer, IMPUTER_PATH)


def main() -> None:
    """Main function to orchestrate the preprocessing pipeline."""
    raw_data = load_data()
    train_data, val_data, test_data = split_data(raw_data)
    train_processed, val_processed, test_processed, imputer = preprocess_data(
        train_data, val_data, test_data
    )
    save_artifacts(train_processed, val_processed, test_processed, imputer)
    logger.info("Data preprocessing completed")


if __name__ == "__main__":
    main()
