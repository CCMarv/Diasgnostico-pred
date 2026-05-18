from __future__ import annotations

import logging
from typing import Final

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

_LOG = logging.getLogger(__name__)


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

    def __init__(self, use_knn: bool = False, use_smote: bool = False, smote_kwargs: dict | None = None) -> None:
        """
        Parámetros opcionales:
        - `use_knn`: si True, usa `KNNImputer` para variables continuas en lugar de `SimpleImputer(median)`.
        - `use_smote`: si True, los métodos `construir_pipeline` retornarán una `imblearn.pipeline.Pipeline` que incluye `SMOTE`.
        - `smote_kwargs`: argumentos pasados a `SMOTE(...)` si `use_smote` es True.
        """
        self.use_knn = use_knn
        self.use_smote = use_smote
        self.smote_kwargs = smote_kwargs or {}

    def construir(self) -> ColumnTransformer:
        """
        Propósito:
        Crear el ColumnTransformer base para columnas CDC.
        """
        # Las columnas se separan por tipo para tratar cada una con la transformación correcta.
        # Esto evita escalar variables binarias y conserva el orden clínico de las ordinales.
        # Elegir imputador de continuas; por defecto SimpleImputer(median), opcionalmente KNNImputer.
        imputador_continuas = KNNImputer() if getattr(self, "use_knn", False) else SimpleImputer(strategy="median")
        continuas = Pipeline(
            steps=[
                ("imputer", imputador_continuas),
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
        # El Pipeline empaqueta transformación + modelo en un solo artefacto serializable.
        # Así validación e inferencia ven exactamente el mismo flujo que entrenamiento.
        if getattr(self, "use_smote", False):
            # Importar de forma local para evitar ImportError si imbalanced-learn no está instalado.
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.pipeline import Pipeline as ImbPipeline

                resampler = SMOTE(**self.smote_kwargs)
                return ImbPipeline(steps=[("preprocesador", self.construir()), ("resample", resampler), ("clasificador", clasificador)])
            except ImportError:
                _LOG.warning(
                    "imbalanced-learn no está instalado. SMOTE desactivado. "
                    "Instala con: pip install imbalanced-learn>=0.12.0"
                )
                return Pipeline(memory=None, steps=[("preprocesador", self.construir()), ("clasificador", clasificador)])
        return Pipeline(memory=None, steps=[("preprocesador", self.construir()), ("clasificador", clasificador)])

    def construir_pipeline_con_fenotipo(self, clasificador) -> Pipeline:
        """
        Propósito:
        Retornar pipeline que añade la columna ordinal de fenotipo.
        """
        # Variante explícita para experimentos donde el fenotipo se añade al contrato de entrada.
        # Se deja separada para no contaminar el pipeline base cuando esa columna no exista.
        continuas = Pipeline(
            steps=[
                ("imputer", KNNImputer() if getattr(self, "use_knn", False) else SimpleImputer(strategy="median")),
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
            memory=None,
            steps=[
                ("preprocesador", preprocesador),
                ("clasificador", clasificador),
            ]
        )
