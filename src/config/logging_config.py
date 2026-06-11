"""Configuração centralizada de logging usando Loguru."""

import os
import sys
from pathlib import Path

from loguru import logger

from src.config.paths import LOGS_DIR


def setup_logger(
    log_level: str = "INFO", log_to_file: bool | None = None, log_dir: Path = LOGS_DIR
) -> None:
    """
    Configura o Loguru para todo o projeto.

    Args:
        log_level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Se deve salvar logs em arquivo. Default: env LOG_TO_FILE
            (em containers, LOG_TO_FILE=false — logs vão só para stdout, e a
            plataforma coleta; padrão 12-factor)
        log_dir: Diretório para salvar arquivos de log (default: LOGS_DIR do projeto)
    """
    if log_to_file is None:
        log_to_file = os.getenv("LOG_TO_FILE", "true").lower() != "false"
    # Remove handler padrão
    logger.remove()

    # Handler para console (colorido e formatado)
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    # Handler para arquivo (se habilitado)
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Arquivo geral - rotação diária
        logger.add(
            log_path / "mlops_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            level="DEBUG",
            rotation="00:00",
            retention="30 days",
            compression="zip",
        )

        # Arquivo só de erros
        logger.add(
            log_path / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
            level="ERROR",
            rotation="00:00",
            retention="90 days",
            backtrace=True,
            diagnose=True,
        )


def get_logger():
    """Retorna a instância do logger."""
    return logger
