"""Gerenciamento centralizado de caminhos do projeto.

Todos os caminhos derivam de PROJECT_ROOT (a localização deste arquivo),
então qualquer estágio funciona independentemente do current working directory.
"""

from pathlib import Path

# Root do projeto (independente de onde roda)
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Diretórios principais
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = PROJECT_ROOT / "metrics"
LOGS_DIR = PROJECT_ROOT / "logs"

# Subdiretorios de data
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PREPROCESSED_DIR = DATA_DIR / "preprocessed"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Arquivos de dados
RAW_DATA_PATH = DATA_RAW_DIR / "raw.csv"
TRAIN_PREPROCESSED_PATH = DATA_PREPROCESSED_DIR / "train_preprocessed.csv"
VAL_PREPROCESSED_PATH = DATA_PREPROCESSED_DIR / "val_preprocessed.csv"
TEST_PREPROCESSED_PATH = DATA_PREPROCESSED_DIR / "test_preprocessed.csv"
TRAIN_PROCESSED_PATH = DATA_PROCESSED_DIR / "train_processed.csv"
VAL_PROCESSED_PATH = DATA_PROCESSED_DIR / "val_processed.csv"
TEST_PROCESSED_PATH = DATA_PROCESSED_DIR / "test_processed.csv"

# Configuração
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# Artefatos de preprocessing (sem colchetes: quebram glob, shell e URLs)
IMPUTER_PATH = ARTIFACTS_DIR / "features_mean_imputer.joblib"
SCALER_PATH = ARTIFACTS_DIR / "features_scaler.joblib"

# Modelo e métricas
MODEL_PATH = MODELS_DIR / "model.keras"
MLFLOW_RUN_ID_PATH = ARTIFACTS_DIR / "mlflow_run_id.txt"
TRAINING_METRICS_PATH = METRICS_DIR / "training.json"
EVALUATION_METRICS_PATH = METRICS_DIR / "evaluation.json"


# Criar diretórios se não existirem
def setup_directories() -> None:
    """Cria todos os diretórios necessários."""
    for directory in [
        DATA_DIR,
        DATA_RAW_DIR,
        DATA_PREPROCESSED_DIR,
        DATA_PROCESSED_DIR,
        MODELS_DIR,
        ARTIFACTS_DIR,
        METRICS_DIR,
        LOGS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
