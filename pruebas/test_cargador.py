from __future__ import annotations

import pandas as pd
import pytest

from config import COLUMNA_OBJETIVO, COLUMNAS_CDC
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


def test_detectar_desbalance_retorna_ratio_correcto():
    y = pd.Series([0] * 860 + [1] * 140, name=COLUMNA_OBJETIVO)
    resultado = CargadorDatos().detectar_desbalance(y)

    assert resultado["ratio"] == pytest.approx(860 / 140, rel=1e-2)
    assert resultado["pct_clase_1"] == pytest.approx(0.14, rel=1e-2)
    assert resultado["recomendacion"] == "class_weight"


def test_analizar_distribucion_incluye_pct_positivos_para_binarias():
    dataframe = pd.DataFrame(
        {
            **{columna: [0, 0, 0, 0, 0] for columna in COLUMNAS_CDC},
            COLUMNA_OBJETIVO: [0, 1, 0, 1, 0],
        }
    )
    dataframe["HighBP"] = [0, 1, 1, 1, 0]

    resultado = CargadorDatos().analizar_distribucion(dataframe)
    assert resultado["HighBP"]["pct_positivos"] == pytest.approx(0.6, rel=1e-3)


def test_cargar_con_objetivo_retorna_x_e_y_separados(tmp_path):
    ruta = tmp_path / "dataset.csv"
    fila = {columna: 1 for columna in COLUMNAS_CDC}
    fila[COLUMNA_OBJETIVO] = 0
    pd.DataFrame([fila]).to_csv(ruta, index=False)

    x, y = CargadorDatos().cargar_con_objetivo(ruta_dataset=ruta)

    assert list(x.columns) == list(COLUMNAS_CDC)
    assert y.name == COLUMNA_OBJETIVO


def test_persistir_procesado_guarda_parquet(tmp_path):
    ruta = tmp_path / "procesado.parquet"
    dataframe = pd.DataFrame([{columna: 1 for columna in COLUMNAS_CDC}])
    salida = CargadorDatos().persistir_procesado(dataframe, ruta_destino=ruta)

    assert salida == ruta
    assert salida.exists()
