from __future__ import annotations

from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class ConstructorPreprocesador:
    """
    Propósito:
    Construir preprocesadores serializables por tipo de columna CDC.
    """

    COLUMNAS_CONTINUAS: Final[list[str]] = ["BMI", "MentHlth", "PhysHlth"]
    COLUMNAS_BINARIAS: Final[list[str]] = [
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
    COLUMNAS_ORDINALES: Final[list[str]] = ["GenHlth", "Age", "Education", "Income"]
    ORDENES_ORDINALES: Final[dict[str, list[int]]] = {
        "GenHlth": [1, 2, 3, 4, 5],
        "Age": list(range(1, 14)),
        "Education": list(range(1, 7)),
        "Income": list(range(1, 9)),
    }
    COLUMNA_FENOTIPO: Final[str] = "fenotipo"

    def construir(self) -> ColumnTransformer:
        """
        Propósito:
        Crear el ColumnTransformer base para columnas CDC.
        """
        continuas = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        binarias = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("no_scaling", "passthrough"),
            ]
        )
        ordinales = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[self.ORDENES_ORDINALES[col] for col in self.COLUMNAS_ORDINALES]
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
        )

    def construir_pipeline(self, clasificador) -> Pipeline:
        """
        Propósito:
        Retornar pipeline completo serializable (preprocesador + clasificador).
        """
        return Pipeline(
            steps=[
                ("preprocesador", self.construir()),
                ("clasificador", clasificador),
            ]
        )

    def construir_pipeline_con_fenotipo(self, clasificador) -> Pipeline:
        """
        Propósito:
        Retornar pipeline que añade la columna ordinal de fenotipo.
        """
        continuas = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        binarias = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("no_scaling", "passthrough"),
            ]
        )
        ordinales = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=[self.ORDENES_ORDINALES[col] for col in self.COLUMNAS_ORDINALES]
                    ),
                ),
            ]
        )
        fenotipo = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
        preprocesador = ColumnTransformer(
            transformers=[
                ("continuas", continuas, self.COLUMNAS_CONTINUAS),
                ("binarias", binarias, self.COLUMNAS_BINARIAS),
                ("ordinales", ordinales, self.COLUMNAS_ORDINALES),
                ("fenotipo", fenotipo, [self.COLUMNA_FENOTIPO]),
            ],
            remainder="drop",
        )
        return Pipeline(
            steps=[
                ("preprocesador", preprocesador),
                ("clasificador", clasificador),
            ]
        )
