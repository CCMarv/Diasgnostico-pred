from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import COLUMNA_OBJETIVO, COLUMNAS_CDC, DATOS_PROCESADOS_DIR, RUTA_DATASET_PREDETERMINADA


class CargadorDatos:
    """Carga, limpia y resume dataset CDC para entrenamiento reproducible."""

    def cargar(self, ruta_dataset: Path | None = None, incluir_objetivo: bool = False) -> pd.DataFrame:
        ruta = ruta_dataset or RUTA_DATASET_PREDETERMINADA
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el dataset en: {ruta}")

        dataframe = pd.read_csv(ruta)
        columnas_requeridas = list(COLUMNAS_CDC)
        if incluir_objetivo:
            columnas_requeridas.append(COLUMNA_OBJETIVO)

        faltantes = [col for col in columnas_requeridas if col not in dataframe.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas requeridas en dataset: {faltantes}")

        return dataframe[columnas_requeridas].copy()

    def limpieza_basica(self, dataframe: pd.DataFrame, incluir_objetivo: bool = False) -> pd.DataFrame:
        columnas_requeridas = list(COLUMNAS_CDC)
        if incluir_objetivo:
            columnas_requeridas.append(COLUMNA_OBJETIVO)

        faltantes = [col for col in columnas_requeridas if col not in dataframe.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas CDC requeridas: {faltantes}")

        limpio = dataframe[columnas_requeridas].apply(pd.to_numeric, errors="coerce").dropna()
        return limpio.reset_index(drop=True)

    def resumen_faltantes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        faltantes = dataframe.isna().sum()
        porcentaje = (faltantes / len(dataframe)).fillna(0.0)
        return pd.DataFrame(
            {
                "columna": faltantes.index,
                "faltantes": faltantes.values,
                "porcentaje": porcentaje.values,
            }
        ).sort_values(by="faltantes", ascending=False)

    def analizar_desbalance(self, dataframe: pd.DataFrame, columna_objetivo: str = COLUMNA_OBJETIVO) -> dict[str, float]:
        if columna_objetivo not in dataframe.columns:
            raise ValueError(f"No se encontró la columna objetivo: {columna_objetivo}")

        distribucion = dataframe[columna_objetivo].value_counts(normalize=True).sort_index()
        clase_0 = float(distribucion.get(0, 0.0))
        clase_1 = float(distribucion.get(1, 0.0))
        ratio = float(clase_0 / clase_1) if clase_1 > 0 else float("inf")

        return {
            "proporcion_clase_0": clase_0,
            "proporcion_clase_1": clase_1,
            "ratio_mayoritaria_vs_minoritaria": ratio,
        }

    def guardar_procesado_parquet(self, dataframe: pd.DataFrame, ruta_salida: Path | None = None) -> Path:
        ruta = ruta_salida or DATOS_PROCESADOS_DIR / "cdc_brfss_2015_limpio.parquet"
        ruta.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(ruta, index=False)
        return ruta
