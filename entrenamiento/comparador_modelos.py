from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from config import SEMILLA_ALEATORIA
from entrenamiento.preprocesador import ConstructorPreprocesador

CLUSTERS_POR_DEFECTO: Final[int] = 3


@dataclass(slots=True)
class ResultadoModelo:
    nombre: str
    puntaje: float
    modelo: object
    metricas: dict[str, float]


class ComparadorModelos:
    """Entrena y compara modelos supervisados y no supervisados para diabetes."""

    def __init__(self, preprocesador: ConstructorPreprocesador | None = None, cv_splits: int = 5) -> None:
        self.preprocesador = preprocesador or ConstructorPreprocesador()
        self.cv_splits = cv_splits

    def entrenar_clasificacion(
        self,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        x_prueba: pd.DataFrame,
        y_prueba: pd.Series,
    ) -> list[ResultadoModelo]:
        resultados = [
            self._entrenar_svm(x_entrenamiento, y_entrenamiento, x_prueba, y_prueba),
            self._entrenar_arbol(x_entrenamiento, y_entrenamiento, x_prueba, y_prueba),
            self._entrenar_gradient_boosting(x_entrenamiento, y_entrenamiento, x_prueba, y_prueba),
            self._entrenar_mlp(x_entrenamiento, y_entrenamiento, x_prueba, y_prueba),
        ]
        return resultados

    def _entrenar_svm(
        self,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        x_prueba: pd.DataFrame,
        y_prueba: pd.Series,
    ) -> ResultadoModelo:
        base = self.preprocesador.construir_pipeline(
            SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=SEMILLA_ALEATORIA)
        )
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=SEMILLA_ALEATORIA)
        grid = GridSearchCV(
            estimator=base,
            param_grid={
                "clasificador__C": [0.1, 1.0, 10.0],
                "clasificador__gamma": ["scale", "auto"],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(x_entrenamiento, y_entrenamiento)
        y_prob = grid.best_estimator_.predict_proba(x_prueba)[:, 1]
        roc_auc = float(roc_auc_score(y_prueba, y_prob))
        return ResultadoModelo(
            nombre="svm_rbf",
            puntaje=roc_auc,
            modelo=grid.best_estimator_,
            metricas={"roc_auc": roc_auc, "cv_roc_auc": float(grid.best_score_)},
        )

    def _entrenar_arbol(
        self,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        x_prueba: pd.DataFrame,
        y_prueba: pd.Series,
    ) -> ResultadoModelo:
        pipeline = self.preprocesador.construir_pipeline(
            DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEMILLA_ALEATORIA)
        )
        pipeline.fit(x_entrenamiento, y_entrenamiento)
        y_prob = pipeline.predict_proba(x_prueba)[:, 1]
        roc_auc = float(roc_auc_score(y_prueba, y_prob))
        return ResultadoModelo(
            nombre="arbol_decision",
            puntaje=roc_auc,
            modelo=pipeline,
            metricas={"roc_auc": roc_auc},
        )

    def _entrenar_gradient_boosting(
        self,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        x_prueba: pd.DataFrame,
        y_prueba: pd.Series,
    ) -> ResultadoModelo:
        pipeline = self.preprocesador.construir_pipeline(
            GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                random_state=SEMILLA_ALEATORIA,
            )
        )
        pipeline.fit(x_entrenamiento, y_entrenamiento)
        y_prob = pipeline.predict_proba(x_prueba)[:, 1]
        roc_auc = float(roc_auc_score(y_prueba, y_prob))
        return ResultadoModelo(
            nombre="gradient_boosting",
            puntaje=roc_auc,
            modelo=pipeline,
            metricas={"roc_auc": roc_auc},
        )

    def _entrenar_mlp(
        self,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        x_prueba: pd.DataFrame,
        y_prueba: pd.Series,
    ) -> ResultadoModelo:
        pipeline = self.preprocesador.construir_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=SEMILLA_ALEATORIA,
            )
        )
        pesos = compute_sample_weight(class_weight="balanced", y=y_entrenamiento)
        pipeline.fit(x_entrenamiento, y_entrenamiento, clasificador__sample_weight=pesos)
        y_prob = pipeline.predict_proba(x_prueba)[:, 1]
        roc_auc = float(roc_auc_score(y_prueba, y_prob))
        return ResultadoModelo(
            nombre="mlp",
            puntaje=roc_auc,
            modelo=pipeline,
            metricas={"roc_auc": roc_auc},
        )

    def entrenar_clustering(self, x_entrenamiento: np.ndarray, n_clusters: int = CLUSTERS_POR_DEFECTO) -> ResultadoModelo:
        modelo = KMeans(n_clusters=n_clusters, random_state=SEMILLA_ALEATORIA, n_init="auto")
        modelo.fit(x_entrenamiento)
        inercia = float(modelo.inertia_)
        return ResultadoModelo(nombre="kmeans", puntaje=inercia, modelo=modelo, metricas={"inercia": inercia})

    @staticmethod
    def seleccionar_mejor(resultados: list[ResultadoModelo], minimizar: bool = False) -> ResultadoModelo:
        if not resultados:
            raise ValueError("No hay resultados para seleccionar el mejor modelo.")
        key_fn = lambda r: r.puntaje
        return min(resultados, key=key_fn) if minimizar else max(resultados, key=key_fn)
