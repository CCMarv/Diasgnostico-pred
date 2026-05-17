from __future__ import annotations

"""Optimizador de hiperparámetros (ítem I5).

Provee una clase `OptimizadorHiperparametros` que envuelve `GridSearchCV`
con `StratifiedKFold` (shuffle=True) usando la semilla de `config.SEMILLA_ALEATORIA`.
"""

from typing import Any

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from config import SEMILLA_ALEATORIA


class OptimizadorHiperparametros:
    """Clase auxiliar para buscar hiperparámetros con GridSearchCV.

    Args:
        cv_splits: número de pliegues en la validación cruzada.
        scoring: métrica para optimizar.
        n_jobs: paralelismo para GridSearchCV.
    """

    def __init__(self, cv_splits: int = 5, scoring: str = "roc_auc", n_jobs: int = -1):
        self.cv_splits = int(cv_splits)
        self.scoring = scoring
        self.n_jobs = n_jobs
        self._last_search: GridSearchCV | None = None

    def buscar(self, pipeline: Pipeline, param_grid: dict[str, Any], X_train, y_train) -> GridSearchCV:
        """Ejecuta GridSearchCV sobre el `pipeline` con `param_grid` y datos de entrenamiento.

        Retorna el objeto `GridSearchCV` ya ajustado.
        """
        cv = StratifiedKFold(n_splits=self.cv_splits, shuffle=True, random_state=SEMILLA_ALEATORIA)
        search = GridSearchCV(pipeline, param_grid, scoring=self.scoring, cv=cv, n_jobs=self.n_jobs, refit=True)
        search.fit(X_train, y_train)
        self._last_search = search
        return search

    def mejor_pipeline(self) -> Pipeline:
        """Devuelve el `best_estimator_` de la última búsqueda.

        Lanza `RuntimeError` si no se ha ejecutado `buscar()` previamente.
        """
        if self._last_search is None:
            raise RuntimeError("No hay búsqueda previa. Ejecuta buscar() antes de pedir mejor_pipeline().")
        return self._last_search.best_estimator_
