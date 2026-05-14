from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    COLUMNA_OBJETIVO,
    COLUMNAS_CDC,
    RUTA_DATASET_PREDETERMINADA,
    RUTA_DATASET_PROCESADO,
)

_LOG = logging.getLogger(__name__)
_UMBRAL_DESBALANCE_MODERADO = 5.0
_UMBRAL_DESBALANCE_CRITICO = 10.0


class CargadorDatos:
    """Carga y aplica una limpieza mínima contractual al dataset CDC."""

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

    def limpieza_basica(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        faltantes = [col for col in COLUMNAS_CDC if col not in dataframe.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas CDC requeridas: {faltantes}")

        limpio = dataframe[list(COLUMNAS_CDC)].apply(pd.to_numeric, errors="coerce").dropna()
        return limpio.reset_index(drop=True)

    def analizar_distribucion(self, dataframe: pd.DataFrame) -> dict[str, dict]:
        """
        Propósito:
        Calcular estadísticas descriptivas por variable para EDA.

        Firma:
        - Parámetros: dataframe con columnas CDC y opcionalmente objetivo.
        - Retorno: dict con métricas por columna.
        """
        columnas = list(COLUMNAS_CDC)
        if COLUMNA_OBJETIVO in dataframe.columns:
            columnas.append(COLUMNA_OBJETIVO)

        faltantes = [col for col in COLUMNAS_CDC if col not in dataframe.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas CDC requeridas para análisis: {faltantes}")

        resumen: dict[str, dict] = {}
        for columna in columnas:
            serie = pd.to_numeric(dataframe[columna], errors="coerce")
            nulos = int(serie.isna().sum())
            stats = {
                "media": float(serie.mean()) if not serie.dropna().empty else np.nan,
                "mediana": float(serie.median()) if not serie.dropna().empty else np.nan,
                "std": float(serie.std()) if not serie.dropna().empty else np.nan,
                "min": float(serie.min()) if not serie.dropna().empty else np.nan,
                "max": float(serie.max()) if not serie.dropna().empty else np.nan,
                "nulos": nulos,
                "pct_nulos": float(nulos / len(dataframe)) if len(dataframe) else 0.0,
            }

            valores_no_nulos = serie.dropna()
            es_binaria = not valores_no_nulos.empty and set(valores_no_nulos.unique()).issubset({0.0, 1.0})
            if es_binaria:
                stats["pct_positivos"] = float(valores_no_nulos.mean())

            if columna == COLUMNA_OBJETIVO:
                desbalance = self.detectar_desbalance(valores_no_nulos.astype(int))
                stats["pct_clase_1"] = desbalance["pct_clase_1"]
                stats["ratio_desbalance"] = desbalance["ratio"]

            resumen[columna] = stats
        return resumen

    def detectar_desbalance(self, y: pd.Series) -> dict[str, float | str]:
        """
        Propósito:
        Cuantificar el desbalance de clases de la variable objetivo.
        """
        if y.empty:
            raise ValueError("La serie objetivo está vacía.")

        y_limpia = pd.to_numeric(y, errors="coerce").dropna().astype(int)
        conteos = y_limpia.value_counts()
        n0 = int(conteos.get(0, 0))
        n1 = int(conteos.get(1, 0))
        total = n0 + n1
        if total == 0:
            raise ValueError("No hay clases válidas (0/1) para calcular desbalance.")

        mayoritaria = max(n0, n1)
        minoritaria = min(n0, n1)
        ratio = float(mayoritaria / minoritaria) if minoritaria > 0 else float("inf")

        if ratio > _UMBRAL_DESBALANCE_CRITICO:
            recomendacion = "smote"
        elif ratio > _UMBRAL_DESBALANCE_MODERADO:
            recomendacion = "class_weight"
        else:
            recomendacion = "submuestreo"

        return {
            "ratio": ratio,
            "pct_clase_0": float(n0 / total),
            "pct_clase_1": float(n1 / total),
            "recomendacion": recomendacion,
        }

    def cargar_con_objetivo(self, ruta_dataset: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
        """
        Propósito:
        Retornar X e y listos para train/test split sin alterar contratos previos.
        """
        dataframe = self.cargar(ruta_dataset=ruta_dataset, incluir_objetivo=True)
        x = dataframe[list(COLUMNAS_CDC)].copy()
        y = dataframe[COLUMNA_OBJETIVO].copy()
        return x, y

    def persistir_procesado(self, dataframe: pd.DataFrame, ruta_destino: Path | None = None) -> Path:
        """
        Propósito:
        Guardar un dataset limpio en Parquet comprimido.
        """
        ruta_salida = ruta_destino or RUTA_DATASET_PROCESADO
        ruta_salida.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(ruta_salida, index=False, compression="snappy")
        tam_mb = ruta_salida.stat().st_size / (1024 * 1024)
        _LOG.info("Dataset procesado guardado en %s", ruta_salida)
        _LOG.info("Shape procesado: %s", dataframe.shape)
        _LOG.info("Tamaño archivo Parquet: %.2f MB", tam_mb)
        return ruta_salida
