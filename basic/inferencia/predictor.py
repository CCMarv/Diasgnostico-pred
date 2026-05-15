from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config import COLUMNAS_CDC, RUTA_MODELO_FINAL, VERSION_SISTEMA


class PredictorDiabetes:
    """Encapsula carga y predicción del modelo de riesgo de diabetes."""

    def __init__(self, ruta_modelo: Path | None = None, version: str = VERSION_SISTEMA) -> None:
        self.ruta_modelo = ruta_modelo or RUTA_MODELO_FINAL
        self.version = version
        self._modelo: Any | None = None

    def cargar_modelo(self) -> bool:
        if not self.ruta_modelo.exists():
            self._modelo = None
            return False
        self._modelo = joblib.load(self.ruta_modelo)
        return True

    def esta_listo(self) -> bool:
        return self._modelo is not None

    def predecir(self, entrada: pd.DataFrame) -> dict[str, float | int | str]:
        if self._modelo is None:
            raise FileNotFoundError("Modelo no cargado. Ejecuta cargar_modelo() primero.")
        if entrada.empty or len(entrada) != 1:
            raise ValueError("La entrada debe contener exactamente una fila.")

        faltantes = [col for col in COLUMNAS_CDC if col not in entrada.columns]
        if faltantes:
            raise ValueError(f"Faltan columnas CDC en la entrada: {faltantes}")

        entrada_ordenada = entrada[list(COLUMNAS_CDC)]
        inicio = time.perf_counter()

        if hasattr(self._modelo, "predict_proba"):
            proba_raw = self._modelo.predict_proba(entrada_ordenada)
            probabilidad = float(proba_raw[0][-1])
            clase = int(probabilidad >= 0.5)
        else:
            clase = int(self._modelo.predict(entrada_ordenada)[0])
            probabilidad = float(clase)

        tiempo_ms = int((time.perf_counter() - inicio) * 1000)
        return {
            "probabilidad": probabilidad,
            "clase": clase,
            "version": self.version,
            "tiempo_ms": tiempo_ms,
        }
