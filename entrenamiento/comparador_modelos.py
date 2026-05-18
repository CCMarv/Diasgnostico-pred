from __future__ import annotations

import logging
from dataclasses import dataclass
from collections.abc import Callable
from typing import Final

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from joblib import Parallel, delayed
import threading
import time

from config import MODELOS_SUPERVISADOS, SEMILLA_ALEATORIA
from entrenamiento.preprocesador import ConstructorPreprocesador

_LOG = logging.getLogger(__name__)
CLUSTERS_POR_DEFECTO: Final[int] = 3
_N_SPLITS_VALIDACION: Final[int] = 5


@dataclass(slots=True)
class ResultadoModelo:
    nombre: str
    puntaje: float
    modelo: object


class ComparadorModelos:
    """Comparador de modelos supervisados y clustering para experimentación clínica."""

    def __init__(self, use_knn: bool = False, use_smote: bool = False, smote_kwargs: dict | None = None) -> None:
        """
        Parámetros:
        - `use_knn`: usar `KNNImputer` para continuas si True.
        - `use_smote`: incluir `SMOTE` dentro del `Pipeline` de entrenamiento cuando esté disponible.
        - `smote_kwargs`: argumentos para `SMOTE(...)`.
        """
        self._preprocesador = ConstructorPreprocesador(use_knn=use_knn, use_smote=use_smote, smote_kwargs=smote_kwargs)
        # Catálogo central: aquí se define qué modelos compiten y qué hiperparámetros se exploran.
        # La SVM se registra con GridSearch porque es el modelo del que más depende el ajuste fino.
        self._catalogo_modelos = {
            "svm": {
                "estimador": SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=False,
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
                    ccp_alpha=0.0,
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
        informar_progreso: Callable[[str], None] | None = None,
    ) -> list[ResultadoModelo]:
        # Esta ruta compara todos los supervisados con la misma métrica para que la selección sea justa.
        # El Pipeline se construye por modelo para mantener el preprocesamiento dentro del entrenamiento.
        modelos_objetivo = modelos_a_entrenar or list(MODELOS_SUPERVISADOS)
        desconocidos = [nombre for nombre in modelos_objetivo if nombre not in self._catalogo_modelos]
        if desconocidos:
            raise ValueError(f"Modelos desconocidos solicitados: {desconocidos}")

        cv = StratifiedKFold(n_splits=_N_SPLITS_VALIDACION, shuffle=True, random_state=SEMILLA_ALEATORIA)
        resultados: list[ResultadoModelo] = []

        if informar_progreso is not None:
            informar_progreso(f"Preparando validación cruzada para {len(modelos_objetivo)} modelos")

        total_modelos = len(modelos_objetivo)
        for indice, nombre in enumerate(modelos_objetivo, start=1):
            descriptor = self._catalogo_modelos[nombre]
            estimador = clone(descriptor["estimador"])
            pipeline = self._preprocesador.construir_pipeline(estimador)
            grid = descriptor["grid"]

            if informar_progreso is not None:
                informar_progreso(f"Entrenando {nombre} ({indice}/{total_modelos})")

            if grid:
                resultado = self._entrenar_con_grid(
                    nombre=nombre,
                    grid=grid,
                    cv=cv,
                    x_entrenamiento=x_entrenamiento,
                    y_entrenamiento=y_entrenamiento,
                    informar_progreso=informar_progreso,
                )
            else:
                resultado = self._entrenar_sin_grid(
                    nombre=nombre,
                    pipeline=pipeline,
                    cv=cv,
                    x_entrenamiento=x_entrenamiento,
                    y_entrenamiento=y_entrenamiento,
                    informar_progreso=informar_progreso,
                )

            resultados.append(resultado)
            if informar_progreso is not None:
                informar_progreso(f"Modelo {nombre} completado")

        return resultados

    def _entrenar_con_grid(
        self,
        nombre: str,
        grid: dict[str, list[float | str]],
        cv: StratifiedKFold,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        informar_progreso: Callable[[str], None] | None,
    ) -> ResultadoModelo:
        # La SVM usa una búsqueda manual para evitar la calibración interna de probabilidades dentro de cada fold.
        # Eso reduce mucho el coste y permite registrar el avance de cada combinación.
        combinaciones = list(ParameterGrid(grid))
        mejor_puntaje = float("-inf")
        mejores_parametros: dict[str, float | str] | None = None

        if informar_progreso is not None:
            informar_progreso(f"Buscando hiperparámetros de {nombre} ({len(combinaciones)} combinaciones)")

        for indice, parametros in enumerate(combinaciones, start=1):
            c = float(parametros["clasificador__C"])
            gamma = str(parametros["clasificador__gamma"])
            if informar_progreso is not None:
                informar_progreso(
                    f"{nombre}: probando combinación {indice}/{len(combinaciones)} (C={c}, gamma={gamma})"
                )

            estimador_busqueda = self._crear_svm_estimador(
                c=c,
                gamma=gamma,
                probability=False,
            )
            pipeline_busqueda = self._preprocesador.construir_pipeline(estimador_busqueda)
            puntajes = self._evaluar_por_folds(
                nombre=nombre,
                pipeline=pipeline_busqueda,
                cv=cv,
                x_entrenamiento=x_entrenamiento,
                y_entrenamiento=y_entrenamiento,
                informar_progreso=informar_progreso,
                etiqueta=f"{nombre} C={c} gamma={gamma}",
            )
            promedio = float(np.mean(puntajes))
            _LOG.info(
                "%s combinación %d/%d (C=%s, gamma=%s) -> folds=%s mean=%.4f",
                nombre,
                indice,
                len(combinaciones),
                c,
                gamma,
                np.array2string(np.asarray(puntajes), precision=4, separator=", "),
                promedio,
            )

            if promedio > mejor_puntaje:
                mejor_puntaje = promedio
                mejores_parametros = parametros
                _LOG.info("%s nuevo mejor puntaje: %.4f con %s", nombre, mejor_puntaje, mejores_parametros)

        if mejores_parametros is None:
            raise ValueError(f"No se pudo seleccionar hiperparámetros para {nombre}.")

        if informar_progreso is not None:
            informar_progreso(
                f"Refit final de {nombre} con C={mejores_parametros['clasificador__C']} y gamma={mejores_parametros['clasificador__gamma']}"
            )

        estimador_final = self._crear_svm_estimador(
            c=float(mejores_parametros["clasificador__C"]),
            gamma=str(mejores_parametros["clasificador__gamma"]),
            probability=True,
        )
        pipeline_final = self._preprocesador.construir_pipeline(estimador_final)
        # Refit final: usar todo el dataset para obtener el mejor modelo posible.
        pipeline_final.fit(x_entrenamiento, y_entrenamiento)
        _LOG.info("%s refit final completado con parámetros %s", nombre, mejores_parametros)
        return ResultadoModelo(nombre=nombre, puntaje=mejor_puntaje, modelo=pipeline_final)

    def _entrenar_sin_grid(
        self,
        nombre: str,
        pipeline,
        cv: StratifiedKFold,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        informar_progreso: Callable[[str], None] | None,
    ) -> ResultadoModelo:
        # Cuando no hay grid, se evalúa por validación cruzada manual para poder emitir progreso por fold.
        if informar_progreso is not None:
            informar_progreso(f"Validando {nombre} con {cv.get_n_splits()} folds")
        puntajes = self._evaluar_por_folds(
            nombre=nombre,
            pipeline=pipeline,
            cv=cv,
            x_entrenamiento=x_entrenamiento,
            y_entrenamiento=y_entrenamiento,
            informar_progreso=informar_progreso,
            etiqueta=nombre,
        )
        if informar_progreso is not None:
            informar_progreso(f"Ajustando modelo final {nombre}")
        pipeline.fit(x_entrenamiento, y_entrenamiento)
        return ResultadoModelo(nombre=nombre, puntaje=float(np.mean(puntajes)), modelo=pipeline)

    def _evaluar_por_folds(
        self,
        nombre: str,
        pipeline,
        cv: StratifiedKFold,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        informar_progreso: Callable[[str], None] | None,
        etiqueta: str,
    ) -> list[float]:
        total_folds = cv.get_n_splits()
        particiones = list(cv.split(x_entrenamiento, y_entrenamiento))

        resultados_fold = Parallel(n_jobs=-1, backend="loky")(
            delayed(self._evaluar_un_fold)(
                nombre=nombre,
                pipeline=pipeline,
                x_entrenamiento=x_entrenamiento,
                y_entrenamiento=y_entrenamiento,
                indices_fit=indices_fit,
                indices_validacion=indices_validacion,
                indice_fold=indice_fold,
                total_folds=total_folds,
                etiqueta=etiqueta,
                informar_progreso=informar_progreso,
            )
            for indice_fold, (indices_fit, indices_validacion) in enumerate(particiones, start=1)
        )

        return [puntaje for _, puntaje in sorted(resultados_fold, key=lambda item: item[0])]

    def _evaluar_un_fold(
        self,
        nombre: str,
        pipeline,
        x_entrenamiento: pd.DataFrame,
        y_entrenamiento: pd.Series,
        indices_fit: np.ndarray,
        indices_validacion: np.ndarray,
        indice_fold: int,
        total_folds: int,
        etiqueta: str,
        informar_progreso: Callable[[str], None] | None,
    ) -> tuple[int, float]:
        if informar_progreso is not None:
            informar_progreso(f"{etiqueta}: entrenando fold {indice_fold}/{total_folds}")

        x_fit = x_entrenamiento.iloc[indices_fit]
        y_fit = y_entrenamiento.iloc[indices_fit]
        x_val = x_entrenamiento.iloc[indices_validacion]
        y_val = y_entrenamiento.iloc[indices_validacion]

        modelo_fold = clone(pipeline)

        # Monitor ligero por fold: emite un latido cada 15s mientras dura el fit
        stop_event = threading.Event()

        def _monitor_fold():
            inicio = time.time()
            while not stop_event.wait(15.0):
                elapsed = int(time.time() - inicio)
                msg = f"{etiqueta}: fold {indice_fold}/{total_folds} en progreso (elapsed={elapsed}s)"
                _LOG.info(msg)
                if informar_progreso is not None:
                    informar_progreso(msg)

        monitor_thread = threading.Thread(target=_monitor_fold, name=f"monitor-{nombre}-fold{indice_fold}", daemon=True)
        monitor_thread.start()

        modelo_fold.fit(x_fit, y_fit)
        # indicar al monitor que termine y esperar su cierre
        stop_event.set()
        monitor_thread.join(timeout=1.0)

        y_score = self._extraer_scores_para_auc(modelo_fold, x_val)
        puntaje = float(roc_auc_score(y_val, y_score))
        _LOG.info(
            "%s fold %d/%d -> roc_auc=%.4f",
            nombre,
            indice_fold,
            total_folds,
            puntaje,
        )

        if informar_progreso is not None:
            informar_progreso(f"{etiqueta}: fold {indice_fold}/{total_folds} terminado (roc_auc={puntaje:.4f})")

        return indice_fold, puntaje

    @staticmethod
    def _extraer_scores_para_auc(modelo, x_validacion: pd.DataFrame | np.ndarray) -> np.ndarray:
        if hasattr(modelo, "predict_proba"):
            return np.asarray(modelo.predict_proba(x_validacion), dtype=float)[:, -1]
        if hasattr(modelo, "decision_function"):
            return np.asarray(modelo.decision_function(x_validacion), dtype=float)
        return np.asarray(modelo.predict(x_validacion), dtype=float)

    def entrenar_clustering(self, x_entrenamiento: np.ndarray, n_clusters: int = CLUSTERS_POR_DEFECTO) -> ResultadoModelo:
        # Tarea separada: si el clustering se reencuadra como fenotipado clínico, esta sección debe reflejar esa lógica.
        modelo = KMeans(n_clusters=n_clusters, random_state=SEMILLA_ALEATORIA, n_init="auto")
        modelo.fit(x_entrenamiento)
        inercia = float(modelo.inertia_)
        return ResultadoModelo(nombre="kmeans", puntaje=inercia, modelo=modelo)

    @staticmethod
    def _crear_svm_estimador(c: float, gamma: str, probability: bool) -> SVC:
        return SVC(
            kernel="rbf",
            C=c,
            gamma=gamma,
            probability=probability,
            class_weight="balanced",
            random_state=SEMILLA_ALEATORIA,
        )

    @staticmethod
    def seleccionar_mejor(resultados: list[ResultadoModelo], minimizar: bool = False) -> ResultadoModelo:
        if not resultados:
            raise ValueError("No hay resultados para seleccionar el mejor modelo.")
        key_fn = (lambda r: r.puntaje)
        return min(resultados, key=key_fn) if minimizar else max(resultados, key=key_fn)
