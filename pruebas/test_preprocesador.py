from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import COLUMNAS_CDC
from entrenamiento.preprocesador import ConstructorPreprocesador


def _base_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "HighBP": 0,
                "HighChol": 0,
                "CholCheck": 1,
                "BMI": 10.0,
                "Smoker": 0,
                "Stroke": 0,
                "HeartDiseaseorAttack": 0,
                "PhysActivity": 1,
                "Fruits": 1,
                "Veggies": 1,
                "HvyAlcoholConsump": 0,
                "AnyHealthcare": 1,
                "NoDocbcCost": 0,
                "GenHlth": 1,
                "MentHlth": 0,
                "PhysHlth": 0,
                "DiffWalk": 0,
                "Sex": 0,
                "Age": 2,
                "Education": 2,
                "Income": 2,
            },
            {
                "HighBP": 1,
                "HighChol": 1,
                "CholCheck": 1,
                "BMI": 30.0,
                "Smoker": 1,
                "Stroke": 0,
                "HeartDiseaseorAttack": 0,
                "PhysActivity": 0,
                "Fruits": 0,
                "Veggies": 0,
                "HvyAlcoholConsump": 0,
                "AnyHealthcare": 1,
                "NoDocbcCost": 1,
                "GenHlth": 4,
                "MentHlth": 10,
                "PhysHlth": 8,
                "DiffWalk": 1,
                "Sex": 1,
                "Age": 9,
                "Education": 5,
                "Income": 6,
            },
        ]
    )[list(COLUMNAS_CDC)]


def test_preprocesador_imputa_desde_entrenamiento():
    x_train = _base_dataframe()
    x_test = _base_dataframe().copy()
    x_test.loc[0, "BMI"] = np.nan

    preprocesador = ConstructorPreprocesador().construir()
    preprocesador.fit(x_train)
    transformado = preprocesador.transform(x_test)

    assert transformado.shape[1] == len(COLUMNAS_CDC)
    assert abs(float(transformado[0, 0])) < 1e-8


def test_pipeline_preprocesador_clasificador():
    x = _base_dataframe()
    y = pd.Series([0, 1])

    pipeline = ConstructorPreprocesador().construir_pipeline(
        LogisticRegression(solver="liblinear", random_state=42)
    )
    pipeline.fit(x, y)

    probas = pipeline.predict_proba(x)
    assert probas.shape == (2, 2)
