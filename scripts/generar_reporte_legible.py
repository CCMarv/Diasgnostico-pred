from __future__ import annotations

"""CLI para reconstruir reportes legibles a partir de JSON crudos del pipeline."""

import argparse
import json
from pathlib import Path

import pandas as pd

from entrenamiento.generador_reportes import (
    construir_reporte_clasificacion,
    construir_reporte_clustering,
    guardar_reporte_legible,
    tabla_comparativa_desde_metricas,
    ruta_legible_desde_crudo,
)


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera un reporte Markdown legible desde métricas crudas")
    parser.add_argument("--entrada-cruda", type=Path, required=True)
    parser.add_argument("--salida-legible", type=Path, default=None)
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    datos = json.loads(args.entrada_cruda.read_text(encoding="utf-8"))
    salida = args.salida_legible or ruta_legible_desde_crudo(args.entrada_cruda)

    if "modelos" in datos:
        tabla = tabla_comparativa_desde_metricas(datos)
        reporte = construir_reporte_clasificacion(datos, tabla, ruta_cruda=args.entrada_cruda)
    else:
        reporte = construir_reporte_clustering(datos, ruta_cruda=args.entrada_cruda)

    guardar_reporte_legible(reporte, salida)
    print(f"Reporte legible generado en: {salida}")


if __name__ == "__main__":
    main()