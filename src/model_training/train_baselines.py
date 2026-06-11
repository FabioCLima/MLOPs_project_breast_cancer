"""Baselines clássicos (LogisticRegression, RandomForest) com CV estratificada.

Num dataset de 569 linhas, um único split de validação é ruído: a comparação
honesta entre modelos usa cross-validation. Cada baseline vira uma run no
MLflow, comparável lado a lado com o MLP Keras.
"""

import json

import mlflow
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config.features import TARGET_COLUMN
from src.config.logging_config import setup_logger
from src.config.params import load_params
from src.config.paths import (
    METRICS_DIR,
    TEST_PROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
    VAL_PROCESSED_PATH,
)
from src.config.tracking import setup_mlflow
from src.data_validation.schemas import PROCESSED_SCHEMA, validate

BASELINE_METRICS_PATH = METRICS_DIR / "baselines.json"

# pos_label=0: a classe maligna é a classe de interesse clínico
SCORING = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "malignant_recall": make_scorer(recall_score, pos_label=0),
    "malignant_precision": make_scorer(precision_score, pos_label=0),
}


def build_models(params: dict) -> dict:
    """Instancia os baselines a partir de params.yaml."""
    seed = params["random_seed"]
    lr = params["logistic_regression"]
    rf = params["random_forest"]
    return {
        "logistic_regression": LogisticRegression(
            C=lr["C"], max_iter=lr["max_iter"], random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=rf["n_estimators"], max_depth=rf["max_depth"], random_state=seed
        ),
    }


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Train+val (para CV) e test (para a comparação final)."""
    train = validate(pd.read_csv(TRAIN_PROCESSED_PATH), PROCESSED_SCHEMA, stage="baselines")
    val = validate(pd.read_csv(VAL_PROCESSED_PATH), PROCESSED_SCHEMA, stage="baselines")
    test = validate(pd.read_csv(TEST_PROCESSED_PATH), PROCESSED_SCHEMA, stage="baselines")

    dev = pd.concat([train, val], ignore_index=True)
    X_dev = dev.drop(TARGET_COLUMN, axis=1)
    y_dev = dev[TARGET_COLUMN].astype(int)
    X_test = test.drop(TARGET_COLUMN, axis=1)
    y_test = test[TARGET_COLUMN].astype(int)
    return X_dev, y_dev, X_test, y_test


def evaluate_on_test(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Métricas de teste do modelo ajustado em todo o dev set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "test_accuracy": float((y_pred == y_test).mean()),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_pr_auc": float(average_precision_score(y_test, y_proba)),
        "test_malignant_recall": float(recall_score(y_test, y_pred, pos_label=0)),
        "test_malignant_precision": float(precision_score(y_test, y_pred, pos_label=0)),
    }


def main() -> None:
    setup_logger()
    setup_mlflow()
    params = load_params("baselines")
    X_dev, y_dev, X_test, y_test = load_data()
    cv = StratifiedKFold(
        n_splits=params["cv_folds"], shuffle=True, random_state=params["random_seed"]
    )

    results: dict[str, dict] = {}
    for name, model in build_models(params).items():
        logger.info(f"Cross-validating {name} ({params['cv_folds']} folds)...")
        cv_res = cross_validate(model, X_dev, y_dev, cv=cv, scoring=SCORING)
        cv_metrics = {
            f"cv_{metric}_mean": float(cv_res[f"test_{metric}"].mean()) for metric in SCORING
        } | {f"cv_{metric}_std": float(cv_res[f"test_{metric}"].std()) for metric in SCORING}

        model.fit(X_dev, y_dev)
        test_metrics = evaluate_on_test(model, X_test, y_test)

        with mlflow.start_run(run_name=f"baseline-{name}"):
            mlflow.log_params({"model": name, **_flat_params(params, name)})
            mlflow.log_metrics(cv_metrics | test_metrics)

        results[name] = {"cv": cv_metrics, "test": test_metrics}
        logger.info(
            f"{name}: ROC-AUC(cv) {cv_metrics['cv_roc_auc_mean']:.4f}"
            f"±{cv_metrics['cv_roc_auc_std']:.4f} | "
            f"recall maligno(test) {test_metrics['test_malignant_recall']:.3f}"
        )

    BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Baseline comparison saved to {BASELINE_METRICS_PATH}")


def _flat_params(params: dict, model_name: str) -> dict:
    """Hiperparâmetros do modelo em formato plano para o MLflow."""
    return dict(params[model_name]) | {
        "cv_folds": params["cv_folds"],
        "random_seed": params["random_seed"],
    }


if __name__ == "__main__":
    main()
