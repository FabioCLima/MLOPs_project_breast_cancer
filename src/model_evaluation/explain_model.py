"""Explicabilidade via permutation importance — modelo-agnóstica.

Permutation importance responde: "quanto a métrica degrada se esta feature
virar ruído?". Funciona para qualquer modelo (sklearn ou Keras) porque só
precisa de predições — não de gradientes nem de estrutura interna.

SHAP foi deliberadamente deixado de fora: para 30 features tabulares e
modelos simples, permutation importance + coeficientes da regressão
logística cobrem a necessidade sem uma dependência pesada extra.
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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config.features import TARGET_COLUMN
from src.config.logging_config import setup_logger
from src.config.params import load_params
from src.config.paths import (
    METRICS_DIR,
    MLFLOW_RUN_ID_PATH,
    MODEL_PATH,
    TEST_PROCESSED_PATH,
    TRAIN_PROCESSED_PATH,
    VAL_PROCESSED_PATH,
)
from src.config.tracking import setup_mlflow
from src.data_validation.schemas import PROCESSED_SCHEMA, validate

EXPLAIN_DIR = METRICS_DIR / "explainability"
N_REPEATS = 10
TOP_K = 15


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = validate(pd.read_csv(TRAIN_PROCESSED_PATH), PROCESSED_SCHEMA, stage="explain")
    val = validate(pd.read_csv(VAL_PROCESSED_PATH), PROCESSED_SCHEMA, stage="explain")
    test = validate(pd.read_csv(TEST_PROCESSED_PATH), PROCESSED_SCHEMA, stage="explain")
    dev = pd.concat([train, val], ignore_index=True)
    return (
        dev.drop(TARGET_COLUMN, axis=1),
        dev[TARGET_COLUMN].astype(int),
        test.drop(TARGET_COLUMN, axis=1),
        test[TARGET_COLUMN].astype(int),
    )


def keras_permutation_importance(
    model: tf.keras.Model, X: pd.DataFrame, y: pd.Series, seed: int
) -> pd.DataFrame:
    """Permutation importance manual para o modelo Keras (queda de ROC-AUC)."""
    rng = np.random.default_rng(seed)
    baseline = roc_auc_score(y, model.predict(X.to_numpy(), verbose=0).ravel())

    rows = []
    for col_idx, col in enumerate(X.columns):
        drops = []
        for _ in range(N_REPEATS):
            X_perm = X.to_numpy().copy()
            X_perm[:, col_idx] = rng.permutation(X_perm[:, col_idx])
            auc = roc_auc_score(y, model.predict(X_perm, verbose=0).ravel())
            drops.append(baseline - auc)
        rows.append(
            {
                "feature": col,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def sklearn_importance(
    X_dev: pd.DataFrame, y_dev: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, seed: int
) -> pd.DataFrame:
    """Permutation importance do baseline LogisticRegression (sklearn)."""
    lr_params = load_params("baselines")["logistic_regression"]
    model = LogisticRegression(C=lr_params["C"], max_iter=lr_params["max_iter"], random_state=seed)
    model.fit(X_dev, y_dev)
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=N_REPEATS, random_state=seed
    )
    return (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def plot_importance(df: pd.DataFrame, title: str, path) -> None:
    top = df.head(TOP_K).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set(title=title, xlabel="Queda de ROC-AUC ao permutar a feature")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    setup_logger()
    seed = load_params("baselines")["random_seed"]
    X_dev, y_dev, X_test, y_test = load_data()

    EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Permutation importance — LogisticRegression (sklearn)...")
    lr_imp = sklearn_importance(X_dev, y_dev, X_test, y_test, seed)
    plot_importance(
        lr_imp, "Permutation importance — Logistic Regression", EXPLAIN_DIR / "lr_importance.png"
    )

    logger.info("Permutation importance — Keras MLP (manual)...")
    keras_model = tf.keras.models.load_model(MODEL_PATH)
    keras_imp = keras_permutation_importance(keras_model, X_test, y_test, seed)
    plot_importance(
        keras_imp, "Permutation importance — Keras MLP", EXPLAIN_DIR / "keras_importance.png"
    )

    payload = {
        "logistic_regression": lr_imp.head(TOP_K).to_dict(orient="records"),
        "keras_mlp": keras_imp.head(TOP_K).to_dict(orient="records"),
    }
    with open(EXPLAIN_DIR / "importance.json", "w") as f:
        json.dump(payload, f, indent=2)

    # Anexa à run de treino no MLflow
    setup_mlflow()
    run_id = MLFLOW_RUN_ID_PATH.read_text().strip()
    with mlflow.start_run(run_id=run_id):
        for artifact in EXPLAIN_DIR.iterdir():
            mlflow.log_artifact(str(artifact), artifact_path="explainability")

    top3 = ", ".join(keras_imp.head(3)["feature"])
    logger.success(f"Explainability salvo em {EXPLAIN_DIR} | top-3 (MLP): {top3}")


if __name__ == "__main__":
    main()
