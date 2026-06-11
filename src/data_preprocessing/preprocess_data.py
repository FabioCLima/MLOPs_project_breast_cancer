import logging
import os

import joblib
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

logger = logging.getLogger("src.data_preprocessing.preprocess_data")


def load_data() -> pd.DataFrame:
    """Load the raw data from disk.

    Returns:
        pd.DataFrame: Raw input data
    """
    input_path = "data/raw/raw.csv"
    logger.info(f"Loading raw data from {input_path}")
    data = pd.read_csv(input_path)
    return data


def load_params() -> dict[str, float | int]:
    """Load preprocessing parameters from params.yaml.

    Returns:
        dict[str, Any]: dictionary containing preprocessing parameters.
    """
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    return params["preprocess_data"]


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation and test sets, stratified by target.

    The validation set is carved out here — before any transformer is fitted —
    so that imputer/scaler statistics never see validation or test rows.

    Args:
        data (pd.DataFrame): Input dataset

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and test datasets
    """
    params = load_params()
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
    data_dir = "data/preprocessed"
    logger.info(f"Saving processed data to {data_dir}")

    train_data.to_csv(os.path.join(data_dir, "train_preprocessed.csv"), index=False)
    val_data.to_csv(os.path.join(data_dir, "val_preprocessed.csv"), index=False)
    test_data.to_csv(os.path.join(data_dir, "test_preprocessed.csv"), index=False)

    # Save imputer
    imputer_path = os.path.join("artifacts", "[features]_mean_imputer.joblib")
    logger.info(f"Saving imputer to {imputer_path}")
    joblib.dump(imputer, imputer_path)


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
