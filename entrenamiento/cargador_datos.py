from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import COLUMNA_OBJETIVO, COLUMNAS_CDC, RUTA_DATASET_PREDETERMINADA


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
