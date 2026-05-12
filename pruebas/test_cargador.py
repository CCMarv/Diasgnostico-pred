from __future__ import annotations

import pandas as pd

from config import COLUMNAS_CDC
from entrenamiento.cargador_datos import CargadorDatos


def test_carga_y_limpieza_basica(tmp_path):
    ruta = tmp_path / "dataset.csv"
    fila_ok = {col: 1 for col in COLUMNAS_CDC}
    fila_nan = {col: 1 for col in COLUMNAS_CDC}
    fila_nan["BMI"] = None
    pd.DataFrame([fila_ok, fila_nan]).to_csv(ruta, index=False)

    cargador = CargadorDatos()
    crudo = cargador.cargar(ruta_dataset=ruta)
    limpio = cargador.limpieza_basica(crudo)

    assert list(crudo.columns) == list(COLUMNAS_CDC)
    assert len(limpio) == 1
    assert not limpio.isna().any().any()
