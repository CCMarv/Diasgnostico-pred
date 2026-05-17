from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

from config import SEMILLA_ALEATORIA, REPORTES_DIR


class FenotipadoKMeans:
    """
    Fenotipado mediante K-Means (ítem I4).

    Esta clase implementa utilidades para generar fenotipos poblacionales
    usando K-Means, calcular el coeficiente de silueta y generar el gráfico
    del codo (elbow plot). Cumple el requisito I4 del sprint 3.

    Métodos principales (en español):
    - `ajustar(X)` ajusta el modelo K-Means sobre `X`.
    - `predecir_fenotipo(X)` retorna las etiquetas de cluster.
    - `graficar_codo(ruta_salida)` guarda un PNG con el elbow plot.
    - `calcular_silhouette(X)` devuelve el `silhouette_score`.

    No usa atributos lambda para asegurar compatibilidad con `joblib`.
    """

    def __init__(self, n_clusters_max: int = 10, random_state: Optional[int] = None, n_clusters: Optional[int] = None):
        """Inicializa el fenotipador.

        Args:
            n_clusters_max: k máximo a considerar para graficar el codo.
            random_state: semilla aleatoria; si None usa `config.SEMILLA_ALEATORIA`.
            n_clusters: (opcional) número de clusters por defecto para ajustar.
        """
        self.n_clusters_max = int(n_clusters_max)
        self.random_state = SEMILLA_ALEATORIA if random_state is None else int(random_state)
        self._model: Optional[KMeans] = None
        self._last_inertias: Optional[list[float]] = None
        self._last_data: Optional[np.ndarray] = None
        self.n_clusters: Optional[int] = int(n_clusters) if n_clusters is not None else None

    def ajustar(self, X, n_clusters: int = 3) -> "FenotipadoKMeans":
        """Ajusta un K-Means con `n_clusters` sobre los datos `X`.

        Args:
            X: DataFrame o array-like con las características.
            n_clusters: número de clusters a ajustar (por defecto 3).

        Returns:
            Instancia de `FenotipadoKMeans` ajustada.
        """
        arr = self._to_numpy(X)
        # guardar datos procesados para operaciones posteriores (ej. elbow)
        self._last_data = arr
        # si hay un n_clusters por defecto en la instancia, úsalo
        if self.n_clusters is None:
            self.n_clusters = int(n_clusters)
        self.n_clusters = int(n_clusters)
        self._model = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init="auto", random_state=self.random_state)
        self._model.fit(arr)
        return self

    def predecir_fenotipo(self, X):
        """Devuelve las etiquetas de fenotipo para `X`.

        Lanza `RuntimeError` si no se ha llamado previamente a `ajustar()`.
        """
        if self._model is None:
            raise RuntimeError("Modelo no ajustado. Llama a ajustar() antes de predecir_fenotipo().")
        arr = self._to_numpy(X)
        return self._model.predict(arr)

    def calcular_silhouette(self, X) -> float:
        """Calcula y devuelve el `silhouette_score` para `X` con el modelo ajustado."""
        if self._model is None:
            raise RuntimeError("Modelo no ajustado. Llama a ajustar() antes de calcular_silhouette().")
        arr = self._to_numpy(X)
        if self.n_clusters is None or self.n_clusters < 2:
            return float("nan")
        labels = self._model.predict(arr)
        return float(silhouette_score(arr, labels))

    def graficar_codo(self, ruta_salida: Optional[Path] = None) -> None:
        """Genera y guarda un elbow plot (inercia vs número de clusters).

        Si `ruta_salida` es `None` se guarda en `reportes/fenotipado_elbow.png`.
        """
        # calcular inercia para k=1..n_clusters_max
        if getattr(self, "_last_data", None) is None:
            raise RuntimeError("No hay datos registrados. Llama a ajustar(X) antes de graficar_codo().")
        inertias = []
        Xdata = self._last_data
        max_k = max(1, min(self.n_clusters_max, Xdata.shape[0]))
        for k in range(1, max_k + 1):
            km = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=self.random_state)
            km.fit(Xdata)
            inertias.append(float(km.inertia_))
        self._last_inertias = inertias

        ruta = REPORTES_DIR / "fenotipado_elbow.png" if ruta_salida is None else Path(ruta_salida)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        plt.figure()
        plt.plot(range(1, len(inertias) + 1), inertias, marker="o")
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Inercia (suma de cuadrados intra-cluster)")
        plt.title("Elbow plot — Fenotipado K-Means")
        plt.grid(True)
        plt.savefig(ruta, bbox_inches="tight")
        plt.close()

    def save(self, path: str):
        """Serializa el objeto completo con `joblib`."""
        joblib.dump(self, path)

    # backward-compatible helpers
    def fit(self, X):
        return self.ajustar(X)

    def predict(self, X):
        return self.predecir_fenotipo(X)

    def fit_predict(self, X):
        self.ajustar(X)
        return self.predecir_fenotipo(X)

    def silhouette(self, X):
        return self.calcular_silhouette(X)

    @staticmethod
    def _to_numpy(X):
        if hasattr(X, "values"):
            return np.asarray(X.values)
        return np.asarray(X)
