"""Contrato canônico de features do projeto.

Esta lista é a única fonte de verdade para os nomes e a ordem das 30 features.
Pipeline, validação de schema e serving importam daqui — nenhum deles deve
depender de sklearn.datasets em runtime.
"""

FEATURE_COLUMNS: list[str] = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

TARGET_COLUMN = "target"

# Convenção do dataset (sklearn load_breast_cancer): 0 = maligno, 1 = benigno
TARGET_LABELS: dict[int, str] = {0: "malignant", 1: "benign"}
