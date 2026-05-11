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


def ejecutar_pipeline(modo: str, ruta_dataset: Path | None, ruta_modelo: Path, ruta_reporte: Path, n_clusters: int = CLUSTERS_POR_DEFECTO) -> dict[str, float | str]:
    """Orquesta entrenamiento mínimo para clasificación o clustering."""
    cargador = CargadorDatos()
    comparador = ComparadorModelos()

    if modo == "clasificacion":
        df = cargador.cargar(ruta_dataset=ruta_dataset, incluir_objetivo=True)
        x = df[list(COLUMNAS_CDC)]
        y = df[COLUMNA_OBJETIVO]

        x_ent, x_pru, y_ent, y_pru = train_test_split(
            x,
            y,
            test_size=PROPORCION_PRUEBA,
            random_state=SEMILLA_ALEATORIA,
            stratify=y,
        )
        resultados = comparador.entrenar_clasificacion(
            x_ent.to_numpy(),
            y_ent.to_numpy(),
            x_pru.to_numpy(),
            y_pru.to_numpy(),
        )
        mejor = comparador.seleccionar_mejor(resultados)
        metricas: dict[str, float | str] = {
            "modo": modo,
            "modelo": mejor.nombre,
            "puntaje": mejor.puntaje,
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
