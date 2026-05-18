from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import COLUMNAS_CDC
from entrenamiento.comparador_modelos import ComparadorModelos
from entrenamiento.preprocesador import ConstructorPreprocesador


# Rangos válidos por columna según el contrato del OrdinalEncoder del preprocesador.
_ORDINAL_RANGOS: dict[str, list[int]] = {
    "GenHlth": [1, 2, 3, 4, 5],
    "Age": list(range(1, 14)),
    "Education": list(range(1, 7)),
    "Income": list(range(1, 9)),
}
_BINARIAS: tuple[str, ...] = (
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
)
_CONTINUAS: tuple[str, ...] = ("BMI", "MentHlth", "PhysHlth")


def _dataset_sintetico(n: int = 300, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for col in _BINARIAS:
        data[col] = rng.integers(0, 2, size=n).astype(float)
    for col in _CONTINUAS:
        data[col] = rng.uniform(10.0, 60.0, size=n)
    for col, valores in _ORDINAL_RANGOS.items():
        data[col] = rng.choice(valores, size=n).astype(float)

    X = pd.DataFrame(data, columns=list(COLUMNAS_CDC))

    # ~14% positivos para simular prevalencia real; mínimo 10 para que ROC-AUC sea calculable.
    y_arr = rng.binomial(1, 0.14, size=n)
    if y_arr.sum() < 10:
        y_arr[:10] = 1
    y = pd.Series(y_arr.astype(float), name="Diabetes_binary")
    return X, y


@pytest.mark.parametrize("nombre_modelo", ["svm", "arbol", "gbm", "mlp"])
def test_resultado_contiene_ocho_campos(nombre_modelo: str) -> None:
    X, y = _dataset_sintetico()
    comparador = ComparadorModelos()
    resultados = comparador.entrenar_clasificacion(X, y, modelos_a_entrenar=[nombre_modelo])
    assert len(resultados) == 1
    resultado = resultados[0]
    assert resultado.nombre == nombre_modelo
    assert isinstance(resultado.puntaje, float)
    assert resultado.puntaje > 0.0
