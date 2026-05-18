from __future__ import annotations

"""Generación de reportes legibles a partir de métricas crudas (ítems S3 y S5)."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import REPORTES_DIR


def guardar_json_crudo(datos: dict, ruta_salida: Path) -> Path:
    """Guarda métricas crudas en JSON para su posterior síntesis.

    Args:
        datos: diccionario serializable con la salida bruta del pipeline.
        ruta_salida: ruta donde se guardará el archivo JSON.

    Returns:
        Ruta final del archivo JSON generado.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta_salida


def construir_reporte_clasificacion(metricas: dict, tabla: pd.DataFrame, ruta_cruda: Path | None = None) -> str:
    """Convierte las métricas crudas de clasificación en un reporte legible.

    Args:
        metricas: salida cruda serializable generada por el pipeline.
        tabla: tabla comparativa de modelos.
        ruta_cruda: ruta del JSON crudo para documentar el origen.

    Returns:
        Texto Markdown listo para persistirse como reporte humano.
    """
    timestamp = metricas.get("timestamp", datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S"))
    mejor_modelo = metricas.get("mejor_modelo", "desconocido")
    mejor_por_pr = metricas.get("mejor_modelo_por_pr_auc", mejor_modelo)
    modelos = metricas.get("modelos", {})
    desbalance = metricas.get("desbalance", {})
    n_muestras = metricas.get("n_muestras", "—")
    use_knn = metricas.get("use_knn", True)
    use_smote = metricas.get("use_smote", True)
    semilla = metricas.get("semilla", 42)
    nombres_modelos = ",".join(modelos.keys())

    mejor = modelos.get(mejor_modelo, {})
    lineas: list[str] = [
        "# Reporte de clasificación — Diagnóstico predictivo de diabetes",
        "",
        f"**Fecha:** {timestamp}  ",
        f"**Parámetros:** n={n_muestras} | KNN={use_knn} | SMOTE={use_smote} | semilla={semilla}  ",
        f"**Modelo ganador (ROC-AUC):** `{mejor_modelo}`  ",
        f"**Mejor por PR-AUC:** `{mejor_por_pr}`  ",
    ]
    if ruta_cruda is not None:
        lineas.append(f"**Origen crudo:** `{ruta_cruda.name}`  ")
    lineas.extend([
        "",
        "---",
        "",
        "## Resumen ejecutivo",
        "",
        f"- Se compararon **{len(modelos)} modelos** supervisados sobre el mismo conjunto de prueba sin data leakage.",
        f"- Mejor ROC-AUC: **{mejor.get('roc_auc', 0.0):.4f}** (`{mejor_modelo}`)",
        f"- Mejor PR-AUC: **{modelos.get(mejor_por_pr, {}).get('pr_auc', 0.0):.4f}** (`{mejor_por_pr}`)",
        f"- Prevalencia de diabetes en el conjunto: **{desbalance.get('pct_clase_1', 0.0):.1%}** (ratio {desbalance.get('ratio', 0.0):.1f}:1)",
        "",
        "## Tabla comparativa",
        "",
        _tabla_markdown_con_ganador(tabla, mejor_modelo),
        "",
        "## Interpretación clínica",
        "",
        _interpretar_modelo_ganador(mejor_modelo, mejor),
        "",
        "---",
        "",
        "*Generado automáticamente por `entrenamiento/pipeline.py`*  ",
        f"*Para reproducir: `python -m entrenamiento.pipeline --modo clasificacion --modelos {nombres_modelos}`*",
        "",
    ])
    return "\n".join(lineas)


def tabla_comparativa_desde_metricas(metricas: dict) -> pd.DataFrame:
    """Construye una tabla compacta de métricas a partir del bloque `modelos`.

    Args:
        metricas: diccionario crudo del pipeline con el bloque `modelos`.

    Returns:
        DataFrame con las columnas útiles para lectura humana.
    """
    filas = []
    for nombre, resumen in metricas.get("modelos", {}).items():
        filas.append(
            {
                "nombre_modelo": resumen.get("nombre_modelo", nombre),
                "roc_auc": resumen.get("roc_auc", 0.0),
                "pr_auc": resumen.get("pr_auc", 0.0),
                "sensibilidad": resumen.get("sensibilidad", 0.0),
                "especificidad": resumen.get("especificidad", 0.0),
                "f1_clase_positiva": resumen.get("f1_clase_positiva", 0.0),
                "brier_score": resumen.get("brier_score", 0.0),
                "accuracy": resumen.get("accuracy", 0.0),
            }
        )
    return pd.DataFrame(filas)


def construir_reporte_clustering(metricas: dict, ruta_cruda: Path | None = None) -> str:
    """Convierte las métricas crudas de clustering en un reporte legible.

    Args:
        metricas: salida cruda del modo clustering.
        ruta_cruda: ruta del JSON crudo para documentar el origen.

    Returns:
        Texto Markdown listo para persistirse como reporte humano.
    """
    timestamp = metricas.get("timestamp", datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S"))
    lineas = [
        "# Reporte legible de clustering",
        "",
        f"**Fecha de generación:** {timestamp}",
        f"**Modelo:** {metricas.get('modelo', 'kmeans')}",
    ]
    if ruta_cruda is not None:
        lineas.append(f"**Origen crudo:** {ruta_cruda.as_posix()}")
    lineas.extend([
        "",
        "## Resumen ejecutivo",
        f"- El modelo de clustering obtuvo una inercia de **{metricas.get('puntaje', 0.0):.4f}**.",
        "- Este valor debe interpretarse junto con la coherencia clínica de los fenotipos y no como una métrica supervisada.",
        "",
        "## Nota operativa",
        "El reporte humano se genera a partir del JSON crudo del pipeline y puede reproducirse sin reentrenar el modelo.",
        "",
    ])
    return "\n".join(lineas)


def guardar_reporte_legible(texto: str, ruta_salida: Path) -> Path:
    """Persiste el reporte Markdown legible."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(texto, encoding="utf-8")
    return ruta_salida


def ruta_legible_desde_crudo(ruta_cruda: Path) -> Path:
    """Deriva una ruta Markdown a partir de una ruta JSON cruda."""
    return ruta_cruda.with_suffix(".md")


def _tabla_markdown(tabla: pd.DataFrame) -> str:
    return _tabla_markdown_con_ganador(tabla, ganador=None)


def _tabla_markdown_con_ganador(tabla: pd.DataFrame, ganador: str | None) -> str:
    col_orden = "roc_auc" if "roc_auc" in tabla.columns else tabla.columns[0]
    tabla = tabla.sort_values(col_orden, ascending=False).reset_index(drop=True)
    encabezados = list(tabla.columns)
    lineas = [
        "| " + " | ".join(encabezados) + " |",
        "| " + " | ".join(["---"] * len(encabezados)) + " |",
    ]
    for _, fila in tabla.iterrows():
        nombre = str(fila.get("nombre_modelo", ""))
        es_ganador = ganador is not None and nombre == ganador
        valores = []
        for columna in encabezados:
            valor = fila[columna]
            celda = f"{valor:.4f}" if isinstance(valor, float) else str(valor)
            if es_ganador:
                celda = f"**{celda}**"
            valores.append(celda)
        prefijo = "→ " if es_ganador else ""
        lineas.append("| " + prefijo + " | ".join(valores) + " |")
    return "\n".join(lineas)


def _interpretar_modelo_ganador(nombre_modelo: str, metricas: dict) -> str:
    roc_auc = float(metricas.get("roc_auc", 0.0))
    pr_auc = float(metricas.get("pr_auc", 0.0))
    sensibilidad = float(metricas.get("sensibilidad", 0.0))
    especificidad = float(metricas.get("especificidad", 0.0))
    brier = float(metricas.get("brier_score", 0.0))

    return (
        f"El modelo **{nombre_modelo}** concentra el mejor balance observado: ROC-AUC {roc_auc:.4f}, PR-AUC {pr_auc:.4f}, "
        f"sensibilidad {sensibilidad:.4f}, especificidad {especificidad:.4f} y Brier Score {brier:.4f}. "
        "En un contexto de tamizaje clínico, esto sugiere que el modelo ordena correctamente el riesgo y mantiene una calibración razonable para priorización, aunque la sensibilidad debe revisarse antes de un despliegue operativo."
    )