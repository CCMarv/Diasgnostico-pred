from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from config import MODELOS_SUPERVISADOS, SEMILLA_ALEATORIA
from entrenamiento.preprocesador import ConstructorPreprocesador

CLUSTERS_POR_DEFECTO: Final[int] = 3
_N_SPLITS_VALIDACION: Final[int] = 5


@dataclass(slots=True)
class ResultadoModelo:
    nombre: str
    puntaje: float
    modelo: object


class ComparadorModelos:
    """Comparador de modelos supervisados y clustering para experimentación clínica."""

    def __init__(self) -> None:
        self._preprocesador = ConstructorPreprocesador()
        self._catalogo_modelos = {
            "svm": {
                "estimador": SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=SEMILLA_ALEATORIA,
                ),
                "grid": {
                    "clasificador__C": [0.1, 1, 10],
                    "clasificador__gamma": ["scale", "auto"],
                },
            },
            "arbol": {
                "estimador": DecisionTreeClassifier(
                    max_depth=5,
                    class_weight="balanced",
                    random_state=SEMILLA_ALEATORIA,
                ),
                "grid": None,
            },
            "gbm": {
                "estimador": GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    random_state=SEMILLA_ALEATORIA,
                ),
                "grid": None,
            },
            "mlp": {
                "estimador": MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=SEMILLA_ALEATORIA,
                ),
                "grid": None,
            },
        }

    def entrenar_clasificacion(
        self,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        modelos_a_entrenar: list[str] | None = None,
    ) -> list[ResultadoModelo]:
        modelos_objetivo = modelos_a_entrenar or list(MODELOS_SUPERVISADOS)
        desconocidos = [nombre for nombre in modelos_objetivo if nombre not in self._catalogo_modelos]
        if desconocidos:
            raise ValueError(f"Modelos desconocidos solicitados: {desconocidos}")

        cv = StratifiedKFold(n_splits=_N_SPLITS_VALIDACION, shuffle=True, random_state=SEMILLA_ALEATORIA)
        resultados: list[ResultadoModelo] = []

        for nombre in modelos_objetivo:
            descriptor = self._catalogo_modelos[nombre]
            estimador = clone(descriptor["estimador"])
            pipeline = self._preprocesador.construir_pipeline(estimador)
            grid = descriptor["grid"]

            if grid:
                ajuste = GridSearchCV(
                    estimator=pipeline,
                    param_grid=grid,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
                ajuste.fit(x_entrenamiento, y_entrenamiento)
                resultados.append(
                    ResultadoModelo(nombre=nombre, puntaje=float(ajuste.best_score_), modelo=ajuste.best_estimator_)
                )
                continue

            puntajes = cross_val_score(
                pipeline,
                x_entrenamiento,
                y_entrenamiento,
                cv=cv,
                scoring="roc_auc",
                n_jobs=-1,
            )
            pipeline.fit(x_entrenamiento, y_entrenamiento)
            resultados.append(ResultadoModelo(nombre=nombre, puntaje=float(np.mean(puntajes)), modelo=pipeline))

        return resultados

    def entrenar_clustering(self, x_entrenamiento: np.ndarray, n_clusters: int = CLUSTERS_POR_DEFECTO) -> ResultadoModelo:
        modelo = KMeans(n_clusters=n_clusters, random_state=SEMILLA_ALEATORIA, n_init="auto")
        modelo.fit(x_entrenamiento)
        inercia = float(modelo.inertia_)
        return ResultadoModelo(nombre="kmeans", puntaje=inercia, modelo=modelo)

    @staticmethod
    def seleccionar_mejor(resultados: list[ResultadoModelo], minimizar: bool = False) -> ResultadoModelo:
        if not resultados:
            raise ValueError("No hay resultados para seleccionar el mejor modelo.")
        key_fn = (lambda r: r.puntaje)
        return min(resultados, key=key_fn) if minimizar else max(resultados, key=key_fn)
