from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import RUTA_DATASET_PREDETERMINADA, REPORTES_DIR
from entrenamiento.cargador_datos import CargadorDatos

ENSANUT_REFS = {
    "HighBP": 0.301,
    "HighChol": 0.196,
    "Smoker": 0.176,
    "PhysActivity": 0.605,
    "Fruits": 0.420,
    "Veggies": 0.340,
    "HvyAlcoholConsump": 0.076,
}
_UMBRAL_SESGO_BAJO = 10.0
_UMBRAL_SESGO_MEDIO = 30.0


def _tabla_a_markdown(tabla: pd.DataFrame) -> str:
    encabezados = list(tabla.columns)
    lineas = [
        "| " + " | ".join(encabezados) + " |",
        "| " + " | ".join(["---"] * len(encabezados)) + " |",
    ]
    for _, fila in tabla.iterrows():
        valores = []
        for columna in encabezados:
            valor = fila[columna]
            if isinstance(valor, float):
                valores.append(f"{valor:.4f}")
            else:
                valores.append(str(valor))
        lineas.append("| " + " | ".join(valores) + " |")
    return "\n".join(lineas)


def _clasificar_sesgo(sesgo_pct: float) -> str:
    if sesgo_pct < _UMBRAL_SESGO_BAJO:
        return "bajo"
    if sesgo_pct <= _UMBRAL_SESGO_MEDIO:
        return "medio"
    return "alto"


def generar_contraste_regional(
    ruta_dataset: Path | None = None,
    ruta_salida: Path | None = None,
) -> Path:
    """
    Propósito:
    Generar reporte markdown de contraste distribucional CDC vs ENSANUT.
    """
    ruta_csv = ruta_dataset or RUTA_DATASET_PREDETERMINADA
    salida = ruta_salida or (REPORTES_DIR / "contraste_regional.md")

    cargador = CargadorDatos()
    df = cargador.cargar(ruta_dataset=ruta_csv, incluir_objetivo=True)

    filas = []
    for variable, ref in ENSANUT_REFS.items():
        prevalencia_cdc = float(pd.to_numeric(df[variable], errors="coerce").mean())
        diferencia = abs(prevalencia_cdc - ref)
        sesgo_pct = float((diferencia / ref) * 100) if ref else 0.0
        filas.append(
            {
                "variable": variable,
                "prevalencia_CDC_calculada": prevalencia_cdc,
                "prevalencia_ENSANUT_referencia": ref,
                "diferencia_absoluta": diferencia,
                "sesgo_pct": sesgo_pct,
                "clasificacion_sesgo": _clasificar_sesgo(sesgo_pct),
            }
        )

    tabla = pd.DataFrame(filas).sort_values("sesgo_pct", ascending=False).reset_index(drop=True)
    sesgo_alto = tabla[tabla["clasificacion_sesgo"] == "alto"]["variable"].tolist()
    variable_max = tabla.iloc[0]["variable"]
    sesgo_max = float(tabla.iloc[0]["sesgo_pct"])

    markdown = "\n".join(
        [
            "# Contraste regional CDC vs ENSANUT",
            "",
            "## Tabla comparativa",
            "",
            _tabla_a_markdown(tabla),
            "",
            "## Narrativa automática",
            "",
            f"- Variables con sesgo alto respecto a México: {sesgo_alto if sesgo_alto else 'ninguna'}.",
            f"- Variable más sesgada: {variable_max} ({sesgo_max:.1f}% de diferencia).",
            "",
            "## Implicaciones para el modelo",
            "",
            (
                "El entrenamiento con CDC BRFSS puede sub/sobre-representar prevalencias observadas "
                "en ENSANUT; se recomienda recalibración local, monitoreo por subgrupos y validación "
                "externa antes de despliegue clínico."
            ),
        ]
    )

    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(markdown + "\n", encoding="utf-8")
    return salida


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genera contraste regional CDC vs ENSANUT")
    parser.add_argument("--dataset", type=Path, default=RUTA_DATASET_PREDETERMINADA)
    parser.add_argument("--salida", type=Path, default=REPORTES_DIR / "contraste_regional.md")
    return parser


def main() -> None:
    args = construir_parser().parse_args()
    generar_contraste_regional(ruta_dataset=args.dataset, ruta_salida=args.salida)


if __name__ == "__main__":
    main()
