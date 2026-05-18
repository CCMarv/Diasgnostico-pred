from __future__ import annotations

import nbformat
from pathlib import Path


def _codigo_notebook() -> str:
    ruta = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda_regionalizado.ipynb"
    nb = nbformat.read(ruta, as_version=4)
    return "\n".join(cell.get("source", "") for cell in nb.cells if cell.get("cell_type") == "code")


def test_bloque_42_usa_bmi_en_lugar_de_bmi5cat() -> None:
    codigo = _codigo_notebook()
    assert "df['bmi_cat'] = df['BMI'].apply(categorizar_bmi)" in codigo
    assert "df['BMI5CAT']" not in codigo


def test_bloque_62_usa_columna_n_nulos() -> None:
    codigo = _codigo_notebook()
    assert "reporte_nulos['n_nulos']" in codigo
    assert "reporte_nulos['total_nulos']" not in codigo
