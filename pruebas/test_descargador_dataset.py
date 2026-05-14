from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from config import COLUMNA_OBJETIVO, COLUMNAS_CDC
from entrenamiento import descargador_dataset


def _repo_falso(features: pd.DataFrame, targets: pd.DataFrame | pd.Series) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(features=features, targets=targets))


def test_descargar_y_persistir_reutiliza_archivo_existente(tmp_path, monkeypatch):
    ruta = tmp_path / "dataset.csv"
    fila = {columna: 0 for columna in COLUMNAS_CDC}
    fila[COLUMNA_OBJETIVO] = 0
    pd.DataFrame([fila]).to_csv(ruta, index=False)

    monkeypatch.setattr(
        descargador_dataset,
        "fetch_ucirepo",
        lambda **_kwargs: pytest.fail("No debe descargar si el archivo ya existe."),
    )

    resultado = descargador_dataset.descargar_y_persistir(ruta_destino=ruta)
    assert resultado == ruta


def test_descargar_y_persistir_descarga_y_valida(tmp_path, monkeypatch):
    ruta = tmp_path / "dataset.csv"
    monkeypatch.setattr(descargador_dataset, "_MIN_FILAS_VALIDAS", 3)

    features = pd.DataFrame(
        [{columna: indice for columna in COLUMNAS_CDC} for indice in range(4)]
    )
    targets = pd.DataFrame({COLUMNA_OBJETIVO: [0, 1, 0, 1]})
    monkeypatch.setattr(descargador_dataset, "fetch_ucirepo", lambda **_kwargs: _repo_falso(features, targets))

    resultado = descargador_dataset.descargar_y_persistir(ruta_destino=ruta)
    guardado = pd.read_csv(resultado)

    assert resultado == ruta
    assert resultado.exists()
    assert list(guardado.columns) == [*COLUMNAS_CDC, COLUMNA_OBJETIVO]
    assert len(guardado) == 4


def test_descargar_y_persistir_falla_si_faltan_columnas(tmp_path, monkeypatch):
    ruta = tmp_path / "dataset.csv"
    monkeypatch.setattr(descargador_dataset, "_MIN_FILAS_VALIDAS", 1)

    columnas_incompletas = list(COLUMNAS_CDC[:-1])
    features = pd.DataFrame([{columna: 1 for columna in columnas_incompletas} for _ in range(2)])
    targets = pd.Series([0, 1], name=COLUMNA_OBJETIVO)
    monkeypatch.setattr(descargador_dataset, "fetch_ucirepo", lambda **_kwargs: _repo_falso(features, targets))

    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        descargador_dataset.descargar_y_persistir(ruta_destino=ruta)
