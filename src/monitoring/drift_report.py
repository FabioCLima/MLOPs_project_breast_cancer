"""Relatório de data drift (Evidently): produção vs. referência de treino.

Referência: features do treino (escala original, pós-imputação).
Produção:   features registradas pelos logs de predição (JSONL do serving).

Uso:
    python -m src.monitoring.drift_report              # usa logs reais
    python -m src.monitoring.drift_report --simulate   # demo com drift sintético

O modo --simulate desloca as features mais importantes (item 13) em +1.5
desvio-padrão — simula um cenário real (ex.: troca de equipamento de medição)
para demonstrar a detecção sem esperar tráfego de produção.
"""

import argparse
import json

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from loguru import logger

from src.config.features import FEATURE_COLUMNS
from src.config.logging_config import setup_logger
from src.config.paths import LOGS_DIR, METRICS_DIR, TEST_PREPROCESSED_PATH, TRAIN_PREPROCESSED_PATH

DRIFT_DIR = METRICS_DIR / "drift"
PREDICTIONS_LOG_PATH = LOGS_DIR / "predictions" / "predictions.jsonl"

# Features que mais carregam o modelo (item 13) — as primeiras a vigiar
TOP_FEATURES = ["worst area", "worst concave points", "worst texture"]


def load_reference() -> pd.DataFrame:
    """Treino na escala original (pós-imputação) — o 'mundo' que o modelo conhece."""
    return pd.read_csv(TRAIN_PREPROCESSED_PATH)[FEATURE_COLUMNS]


def load_production() -> pd.DataFrame:
    """Features extraídas dos logs de predição do serving."""
    if not PREDICTIONS_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Sem logs de predição em {PREDICTIONS_LOG_PATH}. "
            "Gere tráfego na API ou use --simulate."
        )
    rows = [json.loads(line)["features"] for line in PREDICTIONS_LOG_PATH.read_text().splitlines()]
    return pd.DataFrame(rows)[FEATURE_COLUMNS]


def simulate_drifted_production() -> pd.DataFrame:
    """Teste com as features mais importantes deslocadas em +1.5 std."""
    current = pd.read_csv(TEST_PREPROCESSED_PATH)[FEATURE_COLUMNS].copy()
    reference = load_reference()
    for feature in TOP_FEATURES:
        current[feature] = current[feature] + 1.5 * reference[feature].std()
    return current


def run_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Roda o DataDriftPreset e salva HTML + resumo JSON."""
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    report = Report([DataDriftPreset()])
    result = report.run(current_data=current, reference_data=reference)

    html_path = DRIFT_DIR / "drift_report.html"
    result.save_html(str(html_path))

    payload = json.loads(result.json())
    # ValueDrift usa K-S por coluna: p-value < 0.05 => distribuição mudou
    drifted = [
        m["config"]["column"]
        for m in payload["metrics"]
        if m.get("config", {}).get("type", "").endswith("ValueDrift")
        and isinstance(m["value"], int | float)
        and m["value"] < 0.05
    ]
    summary = {
        "n_features": len(FEATURE_COLUMNS),
        "n_drifted": len(drifted),
        "drifted_features": drifted,
        "report_html": str(html_path),
    }
    with open(DRIFT_DIR / "drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true", help="usa drift sintético para demo")
    args = parser.parse_args()

    reference = load_reference()
    current = simulate_drifted_production() if args.simulate else load_production()
    logger.info(
        f"Comparando produção ({len(current)} linhas{', drift simulado' if args.simulate else ''}) "
        f"contra referência de treino ({len(reference)} linhas)"
    )

    summary = run_report(reference, current)
    if summary["n_drifted"] > 0:
        logger.warning(
            f"DRIFT detectado em {summary['n_drifted']}/{summary['n_features']} features: "
            f"{', '.join(summary['drifted_features'][:5])}"
        )
    else:
        logger.success("Nenhum drift detectado")
    logger.info(f"Relatório: {summary['report_html']}")


if __name__ == "__main__":
    main()
