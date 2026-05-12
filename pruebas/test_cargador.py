from __future__ import annotations

import pandas as pd

from config import COLUMNA_OBJETIVO, COLUMNAS_CDC
from entrenamiento.cargador_datos import CargadorDatos


def test_carga_y_limpieza_basica(tmp_path):
    ruta = tmp_path / "dataset.csv"
    fila_ok = {col: 1 for col in COLUMNAS_CDC}
    fila_ok[COLUMNA_OBJETIVO] = 0
    fila_nan = {col: 1 for col in COLUMNAS_CDC}
    fila_nan[COLUMNA_OBJETIVO] = 1
    fila_nan["BMI"] = None
    pd.DataFrame([fila_ok, fila_nan]).to_csv(ruta, index=False)

    cargador = CargadorDatos()
    crudo = cargador.cargar(ruta_dataset=ruta, incluir_objetivo=True)
    limpio = cargador.limpieza_basica(crudo, incluir_objetivo=True)

    assert list(crudo.columns) == [*COLUMNAS_CDC, COLUMNA_OBJETIVO]
    assert len(limpio) == 1
    assert not limpio.isna().any().any()


def test_analisis_desbalance_retorna_ratio():
    df = pd.DataFrame(
        [
            {**{col: 1 for col in COLUMNAS_CDC}, COLUMNA_OBJETIVO: 0},
            {**{col: 1 for col in COLUMNAS_CDC}, COLUMNA_OBJETIVO: 0},
            {**{col: 1 for col in COLUMNAS_CDC}, COLUMNA_OBJETIVO: 0},
            {**{col: 1 for col in COLUMNAS_CDC}, COLUMNA_OBJETIVO: 1},
        ]
    )

    resumen = CargadorDatos().analizar_desbalance(df)

    assert resumen["proporcion_clase_0"] == 0.75
    assert resumen["proporcion_clase_1"] == 0.25
    assert resumen["ratio_mayoritaria_vs_minoritaria"] == 3.0
