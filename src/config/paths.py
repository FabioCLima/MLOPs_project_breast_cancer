"""Gerenciamento centralizado de caminhos do projeto."""

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

# Arquivos específicos
RAW_DATA_PATH = DATA_RAW_DIR / "raw.csv"
PARAMS_PATH = PROJECT_ROOT / "params.yaml"

# Model paths
MODEL_PATH = MODELS_DIR / "model.keras"


# Criar diretórios se não existirem
def setup_directories():
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
