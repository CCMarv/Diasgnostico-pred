from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class ConstructorPreprocesador:
    """Construye preprocesador reproducible para variables CDC sin data leakage."""

    COLUMNAS_CONTINUAS = ["BMI", "MentHlth", "PhysHlth"]
    COLUMNAS_BINARIAS = [
        "HighBP",
        "HighChol",
        "CholCheck",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "Fruits",
        "Veggies",
        "HvyAlcoholConsump",
        "AnyHealthcare",
        "NoDocbcCost",
        "DiffWalk",
        "Sex",
    ]
    COLUMNAS_ORDINALES = ["GenHlth", "Age", "Education", "Income"]

    CATEGORIAS_ORDINALES = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7, 8],
    ]

    def construir(self) -> ColumnTransformer:
        continuas = Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="median")),
                ("escalador", StandardScaler()),
            ]
        )

        binarias = Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="most_frequent")),
            ]
        )

        ordinales = Pipeline(
            steps=[
                ("imputador", SimpleImputer(strategy="most_frequent")),
                (
                    "codificador",
                    OrdinalEncoder(
                        categories=self.CATEGORIAS_ORDINALES,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("continuas", continuas, self.COLUMNAS_CONTINUAS),
                ("binarias", binarias, self.COLUMNAS_BINARIAS),
                ("ordinales", ordinales, self.COLUMNAS_ORDINALES),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

    def construir_pipeline(self, clasificador: object) -> Pipeline:
        return Pipeline(
            steps=[
                ("preprocesador", self.construir()),
                ("clasificador", clasificador),
            ]
        )
