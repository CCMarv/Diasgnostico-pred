from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from config import (
    COLUMNA_OBJETIVO,
    COLUMNAS_CDC,
    PROPORCION_PRUEBA,
    RUTA_MODELO_FINAL,
    RUTA_REPORTE_METRICAS,
    SEMILLA_ALEATORIA,
)
from entrenamiento.cargador_datos import CargadorDatos
from entrenamiento.comparador_modelos import CLUSTERS_POR_DEFECTO, ComparadorModelos
from entrenamiento.evaluador import EvaluadorClinico


def ejecutar_pipeline(
    modo: str,
    ruta_dataset: Path | None,
    ruta_modelo: Path,
    ruta_reporte: Path,
    n_clusters: int = CLUSTERS_POR_DEFECTO,
) -> dict[str, float | str]:
    """Orquesta entrenamiento clínico para clasificación o clustering."""
    cargador = CargadorDatos()
    comparador = ComparadorModelos()
    evaluador = EvaluadorClinico()

    if modo == "clasificacion":
        df = cargador.cargar(ruta_dataset=ruta_dataset, incluir_objetivo=True)
        limpio = cargador.limpieza_basica(df, incluir_objetivo=True)
        x = limpio[list(COLUMNAS_CDC)]
        y = limpio[COLUMNA_OBJETIVO].astype(int)

        x_ent, x_pru, y_ent, y_pru = train_test_split(
            x,
            y,
            test_size=PROPORCION_PRUEBA,
            random_state=SEMILLA_ALEATORIA,
            stratify=y,
        )
        resultados = comparador.entrenar_clasificacion(
            x_entrenamiento=x_ent,
            y_entrenamiento=y_ent,
            x_prueba=x_pru,
            y_prueba=y_pru,
        )

        for resultado in resultados:
            y_prob = resultado.modelo.predict_proba(x_pru)[:, 1]
            metricas_modelo = evaluador.calcular_metricas(y_pru.to_numpy(), y_prob)
            resultado.metricas.update(metricas_modelo)
            evaluador.graficar_curvas(y_pru.to_numpy(), y_prob, resultado.nombre)

        tabla = evaluador.comparar_modelos(resultados)
        mejor = comparador.seleccionar_mejor(resultados)

        ruta_parquet = cargador.guardar_procesado_parquet(limpio)
        metricas: dict[str, float | str] = {
            "modo": modo,
            "modelo": mejor.nombre,
            "puntaje": float(mejor.puntaje),
            "roc_auc": float(mejor.metricas.get("roc_auc", mejor.puntaje)),
            "sensibilidad": float(mejor.metricas.get("sensibilidad", 0.0)),
            "especificidad": float(mejor.metricas.get("especificidad", 0.0)),
            "f1": float(mejor.metricas.get("f1", 0.0)),
            "pr_auc": float(mejor.metricas.get("pr_auc", 0.0)),
            "brier_score": float(mejor.metricas.get("brier_score", 1.0)),
            "dataset_procesado": str(ruta_parquet),
            "modelos_comparados": ", ".join(tabla["modelo"].astype(str).tolist()),
        }
    elif modo == "clustering":
        df = cargador.cargar(ruta_dataset=ruta_dataset, incluir_objetivo=False)
        limpio = cargador.limpieza_basica(df)
        mejor = comparador.entrenar_clustering(limpio.to_numpy(), n_clusters=n_clusters)
        metricas = {
            "modo": modo,
            "modelo": mejor.nombre,
            "puntaje": mejor.puntaje,
        }
    else:
        raise ValueError("Modo inválido. Usa 'clasificacion' o 'clustering'.")

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mejor.modelo, ruta_modelo)
    ruta_reporte.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    return metricas


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de entrenamiento diabetes-ia-mx")
    parser.add_argument("--modo", choices=["clasificacion", "clustering"], default="clasificacion")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--salida-modelo", type=Path, default=RUTA_MODELO_FINAL)
    parser.add_argument("--salida-reporte", type=Path, default=RUTA_REPORTE_METRICAS)
    parser.add_argument("--clusters", type=int, default=CLUSTERS_POR_DEFECTO)
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    ejecutar_pipeline(
        modo=args.modo,
        ruta_dataset=args.dataset,
        ruta_modelo=args.salida_modelo,
        ruta_reporte=args.salida_reporte,
        n_clusters=args.clusters,
    )


if __name__ == "__main__":
    main()
