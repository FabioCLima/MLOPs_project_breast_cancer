"""Helper centralizado para carregar parâmetros de params.yaml."""

from typing import Any

import yaml

from src.config.paths import PARAMS_PATH


def load_params(section: str | None = None) -> dict[str, Any]:
    """Carrega params.yaml a partir da raiz do projeto.

    Args:
        section: Se informado, retorna apenas essa seção (ex.: "train").

    Returns:
        dict com todos os parâmetros, ou apenas a seção pedida.

    Raises:
        KeyError: se a seção pedida não existir no arquivo.
    """
    with open(PARAMS_PATH) as f:
        params: dict[str, Any] = yaml.safe_load(f)
    if section is not None:
        return params[section]
    return params
