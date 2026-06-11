"""Avaliação do modelo com foco no contexto clínico.

Decisões importantes deste módulo:
- O threshold de decisão é ajustado na VALIDAÇÃO (nunca no teste): maximiza
  o recall da classe maligna sujeito a uma precision mínima.
- O teste é usado uma única vez, com o threshold já fixado.
- Curvas (ROC, PR, calibração, matriz de confusão) são salvas como artefatos
  e anexadas à run de treino no MLflow.
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config.features import TARGET_COLUMN
from src.config.logging_config import setup_logger
from src.config.params import load_params
from src.config.paths import (
    EVALUATION_METRICS_PATH,
    METRICS_DIR,
    MLFLOW_RUN_ID_PATH,
    MODEL_PATH,
    TEST_PROCESSED_PATH,
    VAL_PROCESSED_PATH,
)
from src.config.tracking import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, setup_mlflow
from src.data_validation.schemas import PROCESSED_SCHEMA, validate

PLOTS_DIR = METRICS_DIR / "plots"


def load_model() -> tf.keras.Model:
    """Load the trained Keras model from disk."""
    return tf.keras.models.load_model(MODEL_PATH)


def load_split(path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed split as (features, labels)."""
    logger.info(f"Loading data from {path}")
    data = validate(pd.read_csv(path), PROCESSED_SCHEMA, stage="evaluate")
    X = data.drop(TARGET_COLUMN, axis=1)
    y = data[TARGET_COLUMN].astype(int)
    return X, y


def tune_threshold(
    y_true: pd.Series, proba_benign: np.ndarray, min_malignant_precision: float
) -> float:
    """Escolhe o threshold que maximiza o recall maligno com precision mínima.

    O modelo emite P(benign). Predizemos maligno quando P(benign) < t; subir t
    aumenta o recall maligno (menos falsos negativos) ao custo da precision.

    Args:
        y_true: labels verdadeiros da VALIDAÇÃO.
        proba_benign: P(benign) na validação.
        min_malignant_precision: restrição de precision da classe maligna.

    Returns:
        Threshold escolhido (fallback 0.5 se nenhum candidato satisfaz a restrição).
    """
    best_t, best_recall, best_precision = 0.5, -1.0, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_pred = (proba_benign >= t).astype(int)
        recall_mal = recall_score(y_true, y_pred, pos_label=0)
        precision_mal = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        # Empate em recall: prefere o threshold MAIS ALTO (mais conservador
        # contra falsos negativos em dados novos) — não a maior precision
        if precision_mal >= min_malignant_precision and recall_mal >= best_recall:
            best_t, best_recall, best_precision = float(t), recall_mal, precision_mal

    if best_recall < 0:
        logger.warning(
            f"Nenhum threshold atinge precision maligna >= {min_malignant_precision}; "
            "usando 0.5"
        )
        return 0.5

    logger.info(
        f"Threshold ajustado na validação: {best_t:.2f} "
        f"(recall maligno {best_recall:.3f}, precision {best_precision:.3f})"
    )
    return best_t


def save_plots(y_test: pd.Series, proba_benign: np.ndarray, threshold: float) -> list:
    """Gera ROC, PR (classe maligna), calibração e matriz de confusão normalizada."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    # Para a classe maligna (0): score = 1 - P(benign), label positivo = maligno
    y_mal = 1 - y_test
    score_mal = 1 - proba_benign

    fpr, tpr, _ = roc_curve(y_mal, score_mal)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_mal, score_mal):.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="FPR", ylabel="TPR (recall maligno)", title="ROC — classe maligna")
    ax.legend()
    paths.append(PLOTS_DIR / "roc_curve.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_mal, score_mal)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_mal, score_mal):.3f}")
    ax.set(xlabel="Recall maligno", ylabel="Precision maligna", title="PR — classe maligna")
    ax.legend()
    paths.append(PLOTS_DIR / "pr_curve.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, proba_benign, n_bins=8)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(mean_pred, frac_pos, "o-", label="modelo")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="calibração perfeita")
    ax.set(xlabel="P(benign) prevista", ylabel="Fração benigna real", title="Calibração")
    ax.legend()
    paths.append(PLOTS_DIR / "calibration_curve.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    y_pred = (proba_benign >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center")
    ax.set_xticks([0, 1], ["malignant", "benign"])
    ax.set_yticks([0, 1], ["malignant", "benign"])
    ax.set(xlabel="Predito", ylabel="Real", title=f"Matriz de confusão (t={threshold:.2f})")
    fig.colorbar(im, fraction=0.046)
    paths.append(PLOTS_DIR / "confusion_matrix.png")
    fig.savefig(paths[-1], dpi=120, bbox_inches="tight")
    plt.close(fig)

    return paths


def evaluate_model(
    model: tf.keras.Model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    eval_params: dict,
) -> dict:
    """Tune do threshold na validação e avaliação única no teste."""
    proba_val = model.predict(X_val).ravel()
    threshold = tune_threshold(y_val, proba_val, eval_params["min_malignant_precision"])

    proba_test = model.predict(X_test).ravel()
    y_pred = (proba_test >= threshold).astype(int)

    report = classification_report(
        y_test, y_pred, target_names=["malignant", "benign"], output_dict=True
    )
    evaluation = {
        "decision_threshold": threshold,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "roc_auc_malignant": float(roc_auc_score(1 - y_test, 1 - proba_test)),
        "pr_auc_malignant": float(average_precision_score(1 - y_test, 1 - proba_test)),
        "brier_score": float(brier_score_loss(y_test, proba_test)),
    }

    logger.info(
        "Classification Report (threshold ajustado):\n"
        f"{classification_report(y_test, y_pred, target_names=['malignant', 'benign'])}"
    )
    EVALUATION_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_METRICS_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)

    evaluation["_plots"] = save_plots(y_test, proba_test, threshold)
    return evaluation


def track_and_promote(evaluation: dict, eval_params: dict) -> None:
    """Anexa métricas e plots à run de treino e aplica o gate de promoção."""
    setup_mlflow()
    report = evaluation["classification_report"]
    test_metrics = {
        "test_accuracy": report["accuracy"],
        "test_malignant_recall": report["malignant"]["recall"],
        "test_malignant_precision": report["malignant"]["precision"],
        "test_benign_recall": report["benign"]["recall"],
        "test_f1_macro": report["macro avg"]["f1-score"],
        "test_roc_auc_malignant": evaluation["roc_auc_malignant"],
        "test_pr_auc_malignant": evaluation["pr_auc_malignant"],
        "test_brier_score": evaluation["brier_score"],
        "decision_threshold": evaluation["decision_threshold"],
    }

    run_id = MLFLOW_RUN_ID_PATH.read_text().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(test_metrics)
        for plot in evaluation["_plots"]:
            mlflow.log_artifact(str(plot), artifact_path="evaluation_plots")

    # Gate de promoção: alias "production" só com recall maligno suficiente
    min_recall = eval_params["min_malignant_recall"]
    client = mlflow.MlflowClient()
    latest_version = max(
        int(v.version) for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    )
    if test_metrics["test_malignant_recall"] >= min_recall:
        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, str(latest_version)
        )
        logger.success(
            f"Model v{latest_version} promoted to '{PRODUCTION_ALIAS}' "
            f"(malignant recall {test_metrics['test_malignant_recall']:.3f} >= {min_recall})"
        )
    else:
        logger.warning(
            f"Model v{latest_version} NOT promoted: malignant recall "
            f"{test_metrics['test_malignant_recall']:.3f} < {min_recall}"
        )


def main() -> None:
    """Main function to orchestrate the model evaluation process."""
    setup_logger()
    eval_params = load_params("evaluate")
    model = load_model()
    X_val, y_val = load_split(VAL_PROCESSED_PATH)
    X_test, y_test = load_split(TEST_PROCESSED_PATH)
    evaluation = evaluate_model(model, X_val, y_val, X_test, y_test, eval_params)
    track_and_promote(evaluation, eval_params)
    logger.info("Model evaluation completed")


if __name__ == "__main__":
    main()
