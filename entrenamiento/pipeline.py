from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    MODELOS_SUPERVISADOS,
    PROPORCION_PRUEBA,
    RUTA_MODELO_FINAL,
    RUTA_REPORTE_METRICAS,
    SEMILLA_ALEATORIA,
    RUTA_DATASET_PROCESADO,
    COLUMNA_OBJETIVO,
    COLUMNAS_CDC,
)
from entrenamiento.cargador_datos import CargadorDatos
from entrenamiento.comparador_modelos import CLUSTERS_POR_DEFECTO, ComparadorModelos
from entrenamiento.evaluador import EvaluadorClinico

_LOG = logging.getLogger(__name__)


def _extraer_probabilidad_clase_1(modelo, x) -> list[float]:
    """Normaliza salida del modelo a probabilidades de clase positiva (riesgo de diabetes)."""
    if hasattr(modelo, "predict_proba"):
        # Ruta ideal: la SVM del proyecto ya se entrena con probability=True, así que esta rama debe ser la habitual.
        probabilidades = modelo.predict_proba(x)
        return [float(fila[-1]) for fila in probabilidades]
    if hasattr(modelo, "decision_function"):
        # Si el estimador no expone probabilidades, la decisión se aproxima con una sigmoide.
        decision = np.asarray(modelo.decision_function(x), dtype=float)
        return [float(valor) for valor in (1 / (1 + np.exp(-decision)))]
    pred = modelo.predict(x)
    return [float(valor) for valor in pred]


def _resolver_modelos_a_entrenar(modelos_cli: list[str] | None) -> list[str]:
    """Resuelve la lista final de modelos supervisados a entrenar."""
    if modelos_cli is None:
        return list(MODELOS_SUPERVISADOS)
    return [modelo.strip() for modelo in modelos_cli if modelo.strip()]


def ejecutar_pipeline(
    modo: str,
    ruta_dataset: Path | None,
    ruta_modelo: Path,
    ruta_reporte: Path,
    n_clusters: int = CLUSTERS_POR_DEFECTO,
    modelos_a_entrenar: list[str] | None = None,
) -> dict[str, float | str | dict]:
    """
    Orquesta el flujo completo de entrenamiento.

    Nota para estudiantes:
    - `clasificacion` entrena y compara modelos supervisados.
    - `clustering` ejecuta K-Means sin variable objetivo.
    """
    cargador = CargadorDatos()
    comparador = ComparadorModelos()
    # Configurar logging si aún no hay handlers (útil al ejecutar desde CLI o pruebas aisladas)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    _LOG.info("Iniciando pipeline (modo=%s, ruta_dataset=%s)", modo, ruta_dataset)

    if modo == "clasificacion":
        evaluador = EvaluadorClinico()
        # Se carga el dataset ya con objetivo para poder comparar modelos con la misma partición.
        df = cargador.cargar(ruta_dataset=ruta_dataset, incluir_objetivo=True)
        _LOG.info("Dataset cargado: %s filas, %s columnas", *df.shape)

        # Limpieza básica combinada (X + y): convertir a numérico y eliminar filas con nulos.
        df_limpio = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)

        # Persistir dataset procesado para S2-09 (Parquet comprimido).
        try:
            cargador.persistir_procesado(df_limpio, ruta_destino=RUTA_DATASET_PROCESADO)
            _LOG.info("Dataset procesado persistido en %s", RUTA_DATASET_PROCESADO)
        except Exception as exc:  # pragma: no cover - logging only
            _LOG.warning("No fue posible persistir dataset procesado: %s", exc)

        # Separar X e y a partir del dataframe limpio.
        x = df_limpio[list(COLUMNAS_CDC)].copy()
        y = df_limpio[COLUMNA_OBJETIVO].copy()

        desbalance = cargador.detectar_desbalance(y)
        _LOG.info("Desbalance detectado: %s", desbalance)

        # La separación estratificada preserva la proporción de clases en train y test.
        x_ent, x_pru, y_ent, y_pru = train_test_split(
            x,
            y,
            test_size=PROPORCION_PRUEBA,
            random_state=SEMILLA_ALEATORIA,
            stratify=y,
        )
        _LOG.info("Partición generada: train=%d, test=%d", len(x_ent), len(x_pru))
        modelos_objetivo = _resolver_modelos_a_entrenar(modelos_a_entrenar)
        _LOG.info("Entrenando modelos: %s", ",".join(modelos_objetivo) if modelos_objetivo else "todos")
        resultados = comparador.entrenar_clasificacion(
            x_ent,
            y_ent,
            modelos_a_entrenar=modelos_objetivo,
        )

        evaluaciones = []
        for resultado in resultados:
            _LOG.info("Evaluando modelo: %s", resultado.nombre)
            y_prob = _extraer_probabilidad_clase_1(resultado.modelo, x_pru)
            # Cada modelo se evalúa sobre el mismo conjunto de prueba para comparar ROC-AUC de forma justa.
            evaluacion = evaluador.calcular_metricas(
                y_verdadero=y_pru.to_numpy(),
                y_prob=y_prob,
                nombre_modelo=resultado.nombre,
            )
            try:
                _LOG.info("Resultado %s: ROC-AUC=%.4f", resultado.nombre, evaluacion.roc_auc)
            except Exception:
                _LOG.info("Resultado %s evaluado (no disponible ROC-AUC).", resultado.nombre)
            evaluaciones.append((resultado, evaluacion))

        mejor_resultado, _ = max(evaluaciones, key=lambda item: item[1].roc_auc)
        evaluador.graficar_curvas(y_pru.to_numpy(), _extraer_probabilidad_clase_1(mejor_resultado.modelo, x_pru), mejor_resultado.nombre)
        evaluador.graficar_curva_calibracion(
            y_pru.to_numpy(),
            _extraer_probabilidad_clase_1(mejor_resultado.modelo, x_pru),
            mejor_resultado.nombre,
        )
        evaluador.comparar_modelos([item[1] for item in evaluaciones])

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        ruta_versionada = ruta_modelo.parent / f"modelo_diabetes_v{timestamp}.joblib"

        ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
        ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
        # Se guarda el pipeline completo, no solo el clasificador, para que inferencia reciba el mismo preprocesamiento.
        joblib.dump(mejor_resultado.modelo, ruta_versionada)
        _LOG.info("Modelo versionado guardado en %s", ruta_versionada)
        joblib.dump(mejor_resultado.modelo, ruta_modelo)
        _LOG.info("Modelo final guardado en %s", ruta_modelo)

        modelos_json = {
            resultado.nombre: evaluador.serializar_resultado(evaluacion)
            for resultado, evaluacion in evaluaciones
        }
        metricas: dict[str, float | str | dict] = {
            "version": ruta_versionada.name,
            "timestamp": timestamp,
            "mejor_modelo": mejor_resultado.nombre,
            "modelos": modelos_json,
            "desbalance": desbalance,
            "ruta_modelo_versionado": str(ruta_versionada),
        }
    elif modo == "clustering":
        df = cargador.cargar(ruta_dataset=ruta_dataset, incluir_objetivo=False)
        _LOG.info("Dataset cargado para clustering: %s filas, %s columnas", *df.shape)
        limpio = cargador.limpieza_basica(df)
        _LOG.info("Limpieza básica completada: %d filas", len(limpio))
        mejor = comparador.entrenar_clustering(limpio.to_numpy(), n_clusters=n_clusters)
        _LOG.info("Mejor clustering: %s (puntaje=%.4f)", mejor.nombre, mejor.puntaje)
        ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
        ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(mejor.modelo, ruta_modelo)
        _LOG.info("Modelo clustering guardado en %s", ruta_modelo)
        metricas = {
            "modo": modo,
            "modelo": mejor.nombre,
            "puntaje": mejor.puntaje,
        }
    else:
        raise ValueError("Modo inválido. Usa 'clasificacion' o 'clustering'.")

    ruta_reporte.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    _LOG.info("Reporte de métricas escrito en %s", ruta_reporte)
    return metricas


def construir_parser() -> argparse.ArgumentParser:
    """Construye el parser CLI para ejecutar el pipeline por terminal."""
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento Diasgnostico-pred")
    parser.add_argument("--modo", choices=["clasificacion", "clustering"], default="clasificacion")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--salida-modelo", type=Path, default=RUTA_MODELO_FINAL)
    parser.add_argument("--salida-reporte", type=Path, default=RUTA_REPORTE_METRICAS)
    parser.add_argument("--clusters", type=int, default=CLUSTERS_POR_DEFECTO)
    parser.add_argument(
        "--modelos",
        type=str,
        default=None,
        help="Lista separada por comas de modelos supervisados (ej: gbm,mlp).",
    )
    return parser


def main() -> None:
    """Punto de entrada CLI para entrenamiento reproducible."""
    args = construir_parser().parse_args()
    modelos = args.modelos.split(",") if args.modelos else None
    # Asegurar salida de logs al ejecutar desde CLI
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _LOG.info("Ejecutando desde CLI con args: %s", vars(args))
    ejecutar_pipeline(
        modo=args.modo,
        ruta_dataset=args.dataset,
        ruta_modelo=args.salida_modelo,
        ruta_reporte=args.salida_reporte,
        n_clusters=args.clusters,
        modelos_a_entrenar=modelos,
    )


if __name__ == "__main__":
    main()
