import numpy as np
import pandas as pd
from loguru import logger
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

from src.config.logging_config import setup_logger
from src.config.paths import RAW_DATA_PATH


def fetch_data() -> pd.DataFrame:
    """Fetch the breast cancer dataset and convert to DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing the breast cancer data with features and target
    """
    logger.info("Fetching data...")
    dataset: Bunch = load_breast_cancer()  # type: ignore[assignment]

    # Features columns
    data = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

    # Introduce random NaN values
    np.random.seed(42)
    for col in data.columns:
        mask = np.random.random(len(data)) < 0.05  # 5% chance of NaN
        data.loc[mask, col] = np.nan

    # Target column
    data["target"] = dataset.target

    return data


def save_data(data: pd.DataFrame) -> None:
    """Save the raw data to disk.

    Args:
        data (pd.DataFrame): Raw breast cancer dataset to save
    """
    # Garantir que o diretório existe
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving raw data to {RAW_DATA_PATH}")
    data.to_csv(RAW_DATA_PATH, index=False)
    logger.success(f"Data saved successfully to {RAW_DATA_PATH}")


def main() -> None:
    """Main function to orchestrate the data loading process."""
    setup_logger()
    raw_data = fetch_data()
    save_data(raw_data)
    logger.info("Data loading completed")


if __name__ == "__main__":
    main()
