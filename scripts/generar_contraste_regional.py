"""Genera reportes/contraste_regional.md comparando prevalencias CDC BRFSS vs ENSANUT 2022.

Uso:
    python scripts/generar_contraste_regional.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import COLUMNAS_CDC, RUTA_DATASET_PROCESADO, REPORTES_DIR

# Referencias epidemiológicas ENSANUT 2022 (prevalencia en adultos mexicanos)
ENSANUT_REFS: dict[str, float] = {
    "HighBP": 0.318,
    "HighChol": 0.229,
    "Smoker": 0.158,
    "Stroke": 0.027,
    "HeartDiseaseorAttack": 0.078,
    "PhysActivity": 0.452,
    "Fruits": 0.556,
    "Veggies": 0.548,
    "HvyAlcoholConsump": 0.053,
    "AnyHealthcare": 0.762,
}

VARIABLES_BINARIAS = [c for c in COLUMNAS_CDC if c in ENSANUT_REFS]


def _clasificar_sesgo(sesgo: float) -> str:
    if abs(sesgo) < 10:
        return "✅ Bajo"
    if abs(sesgo) < 30:
        return "🟡 Moderado"
    return "🔴 Alto"


def generar(ruta_salida: Path | None = None) -> Path:
    df = pd.read_parquet(RUTA_DATASET_PROCESADO)
    n = len(df)

    filas = []
    for var in VARIABLES_BINARIAS:
        if var not in df.columns:
            continue
        cdc_prev = float(df[var].mean())
        ensanut_prev = ENSANUT_REFS[var]
        sesgo = (cdc_prev - ensanut_prev) / ensanut_prev * 100
        filas.append({
            "Variable": var,
            "CDC BRFSS 2015 (%)": round(cdc_prev * 100, 1),
            "ENSANUT 2022 (%)": round(ensanut_prev * 100, 1),
            "Sesgo relativo (%)": round(sesgo, 1),
            "Clasificación": _clasificar_sesgo(sesgo),
        })

    tabla = pd.DataFrame(filas).sort_values("Sesgo relativo (%)", key=abs, ascending=False)

    sesgos_altos = tabla[tabla["Clasificación"] == "🔴 Alto"]
    sesgos_mod = tabla[tabla["Clasificación"] == "🟡 Moderado"]
    sesgos_bajos = tabla[tabla["Clasificación"] == "✅ Bajo"]

    lineas = [
        "# Contraste regional CDC BRFSS 2015 vs ENSANUT 2022",
        "",
        f"**Fecha de generación:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d')}  ",
        f"**Registros CDC analizados:** {n:,}  ",
        f"**Variables comparadas:** {len(filas)} indicadores de salud binarios  ",
        "",
        "---",
        "",
        "## Contexto",
        "",
        "El dataset CDC BRFSS 2015 proviene de una muestra de adultos estadounidenses.",
        "Este análisis cuantifica el **sesgo distribucional** entre las prevalencias observadas",
        "en BRFSS 2015 y las reportadas por ENSANUT 2022 para adultos mexicanos, con el fin de",
        "documentar la transferibilidad del modelo a la población objetivo.",
        "",
        "> **Criterio de lectura:** sesgo relativo = (CDC − ENSANUT) / ENSANUT × 100.  ",
        "> Valores positivos indican sobrerepresentación en CDC; negativos, subrepresentación.",
        "",
        "---",
        "",
        "## Tabla de sesgo distribucional",
        "",
        "| Variable | CDC BRFSS 2015 (%) | ENSANUT 2022 (%) | Sesgo relativo (%) | Clasificación |",
        "|----------|-------------------|-----------------|-------------------|---------------|",
    ]

    for _, row in tabla.iterrows():
        lineas.append(
            f"| {row['Variable']} | {row['CDC BRFSS 2015 (%)']:.1f} | "
            f"{row['ENSANUT 2022 (%)']:.1f} | {row['Sesgo relativo (%)']:+.1f} | "
            f"{row['Clasificación']} |"
        )

    lineas += [
        "",
        "---",
        "",
        "## Resumen ejecutivo",
        "",
        f"- **{len(sesgos_altos)}** variables con sesgo alto (>30%): "
        + (", ".join(sesgos_altos["Variable"].tolist()) if len(sesgos_altos) else "ninguna"),
        f"- **{len(sesgos_mod)}** variables con sesgo moderado (10–30%): "
        + (", ".join(sesgos_mod["Variable"].tolist()) if len(sesgos_mod) else "ninguna"),
        f"- **{len(sesgos_bajos)}** variables con sesgo bajo (<10%): transferibilidad adecuada.",
        "",
        "## Implicaciones para el modelo",
        "",
        "Las variables con sesgo alto requieren interpretación cuidadosa al aplicar el modelo",
        "sobre población mexicana: la decisión del modelo puede verse afectada por diferencias",
        "sistemáticas entre la distribución de entrenamiento y la distribución objetivo.",
        "",
        "---",
        "",
        "*Generado automáticamente por `scripts/generar_contraste_regional.py`*",
        "*Fuente ENSANUT: INSP, Encuesta Nacional de Salud y Nutrición 2022*",
    ]

    ruta = ruta_salida or (REPORTES_DIR / "contraste_regional.md")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    print(f"✅ Generado: {ruta}")
    return ruta


if __name__ == "__main__":
    generar()
