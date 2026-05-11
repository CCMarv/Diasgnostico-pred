from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from config import SEMILLA_ALEATORIA

CLUSTERS_POR_DEFECTO: Final[int] = 3


@dataclass(slots=True)
class ResultadoModelo:
    nombre: str
    puntaje: float
    modelo: object


class ComparadorModelos:
    """Comparador mínimo para dejar lista la interfaz de experimentación."""

    def entrenar_clasificacion(self, x_entrenamiento: np.ndarray, y_entrenamiento: np.ndarray, x_prueba: np.ndarray, y_prueba: np.ndarray) -> list[ResultadoModelo]:
        modelo_base = DummyClassifier(strategy="prior", random_state=SEMILLA_ALEATORIA)
        modelo_base.fit(x_entrenamiento, y_entrenamiento)

        pred = modelo_base.predict(x_prueba)
        puntaje = accuracy_score(y_prueba, pred)
        return [ResultadoModelo(nombre="dummy_prior", puntaje=float(puntaje), modelo=modelo_base)]

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
