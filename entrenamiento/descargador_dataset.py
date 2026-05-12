from __future__ import annotations

from collections.abc import Sequence
import logging
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from config import COLUMNA_OBJETIVO, COLUMNAS_CDC, NOMBRE_DATASET_UCI_ID, RUTA_DATASET_PREDETERMINADA

"""
Propósito:
Descargar el dataset CDC BRFSS 2015 desde UCI ML Repository y persistirlo
en datos/brutos/ para uso offline del pipeline.

Firma pública:
- descargar_y_persistir(ruta_destino: Path | None = None) -> Path
"""

_LOG = logging.getLogger(__name__)
_MIN_FILAS_VALIDAS = 200_000


def _validar_columnas(dataframe: pd.DataFrame, columnas_requeridas: Sequence[str]) -> None:
    """
    Propósito:
    Verificar que el dataframe contiene todas las columnas requeridas.

    Error principal:
    - ValueError cuando falta al menos una columna contractual.
    """
    faltantes = [columna for columna in columnas_requeridas if columna not in dataframe.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en dataset descargado: {faltantes}")


def descargar_y_persistir(ruta_destino: Path | None = None) -> Path:
    """
    Propósito:
    Descargar, validar y persistir el dataset CDC BRFSS 2015.

    Lógica resumida:
    - Si el archivo ya existe, retorna la ruta sin re-descargar.
    - Descarga con UCI id=891.
    - Valida columnas CDC + objetivo y tamaño mínimo.
    - Guarda CSV en datos/brutos/.
    """
    ruta_salida = ruta_destino or RUTA_DATASET_PREDETERMINADA
    if ruta_salida.exists():
        _LOG.info("Dataset ya existe en %s; se reutiliza archivo local.", ruta_salida)
        return ruta_salida

    repo = fetch_ucirepo(id=NOMBRE_DATASET_UCI_ID)
    caracteristicas = repo.data.features
    objetivos = repo.data.targets

    if isinstance(objetivos, pd.Series):
        objetivos = objetivos.to_frame(name=COLUMNA_OBJETIVO)
    elif COLUMNA_OBJETIVO not in objetivos.columns and len(objetivos.columns) == 1:
        objetivos = objetivos.rename(columns={objetivos.columns[0]: COLUMNA_OBJETIVO})

    dataframe = pd.concat([caracteristicas, objetivos], axis=1)
    _validar_columnas(dataframe, (*COLUMNAS_CDC, COLUMNA_OBJETIVO))

    if len(dataframe) <= _MIN_FILAS_VALIDAS:
        raise ValueError(
            f"El dataset descargado tiene {len(dataframe)} filas; se esperaban más de {_MIN_FILAS_VALIDAS}."
        )

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(ruta_salida, index=False)

    prevalencia = float(dataframe[COLUMNA_OBJETIVO].mean())
    nulos = dataframe.isna().sum().to_dict()
    _LOG.info("Dataset CDC guardado en %s", ruta_salida)
    _LOG.info("Forma dataset: %s", dataframe.shape)
    _LOG.info("Prevalencia clase positiva (%s): %.4f", COLUMNA_OBJETIVO, prevalencia)
    _LOG.info("Valores nulos por columna: %s", nulos)
    return ruta_salida
