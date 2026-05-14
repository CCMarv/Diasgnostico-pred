from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from config import COLUMNAS_CDC
from entrenamiento.preprocesador import ConstructorPreprocesador
from inferencia.predictor import PredictorDiabetes


def _fila_base(valor_bmi: float = 30.0) -> dict[str, float | int]:
    fila = {
        "HighBP": 1,
        "HighChol": 1,
        "CholCheck": 1,
        "BMI": valor_bmi,
        "Smoker": 0,
        "Stroke": 0,
        "HeartDiseaseorAttack": 0,
        "PhysActivity": 1,
        "Fruits": 1,
        "Veggies": 1,
        "HvyAlcoholConsump": 0,
        "AnyHealthcare": 1,
        "NoDocbcCost": 0,
        "GenHlth": 3,
        "MentHlth": 2,
        "PhysHlth": 2,
        "DiffWalk": 0,
        "Sex": 1,
        "Age": 7,
        "Education": 4,
        "Income": 6,
    }
    return fila


def test_pipeline_no_filtra_estadisticas_de_test():
    constructor = ConstructorPreprocesador()
    pipeline = constructor.construir_pipeline(DummyClassifier(strategy="most_frequent"))

    x_train = pd.DataFrame([_fila_base(20.0), _fila_base(22.0), _fila_base(24.0)])
    y_train = pd.Series([0, 1, 0])
    x_test = pd.DataFrame([_fila_base(60.0), _fila_base(65.0)])

    pipeline.fit(x_train, y_train)
    pipeline.named_steps["preprocesador"].transform(x_test)

    media_scaler_bmi = pipeline.named_steps["preprocesador"].named_transformers_["continuas"].named_steps[
        "scaler"
    ].mean_[0]
    assert media_scaler_bmi == np.mean([20.0, 22.0, 24.0])


def test_pipeline_serializable_y_compatible_con_predictor(tmp_path):
    constructor = ConstructorPreprocesador()
    pipeline = constructor.construir_pipeline(DummyClassifier(strategy="prior"))

    x_train = pd.DataFrame([_fila_base(28.0), _fila_base(31.0)])
    y_train = pd.Series([0, 1])
    pipeline.fit(x_train, y_train)

    ruta_modelo = tmp_path / "modelo.joblib"
    joblib.dump(pipeline, ruta_modelo)

    predictor = PredictorDiabetes(ruta_modelo=ruta_modelo)
    assert predictor.cargar_modelo() is True

    salida = predictor.predecir(pd.DataFrame([_fila_base(29.0)])[list(COLUMNAS_CDC)])
    assert "probabilidad" in salida


def test_columnas_binarias_no_se_escalan():
    constructor = ConstructorPreprocesador()
    preprocesador = constructor.construir()
    x = pd.DataFrame([_fila_base(28.0), _fila_base(30.0)])
    preprocesador.fit(x)

    paso_final = preprocesador.named_transformers_["binarias"].steps[-1][1]
    assert paso_final == "passthrough"


def test_ordinales_mantienen_orden():
    constructor = ConstructorPreprocesador()
    preprocesador = constructor.construir()

    fila_1 = _fila_base(28.0)
    fila_2 = _fila_base(30.0)
    fila_1["GenHlth"] = 1
    fila_2["GenHlth"] = 5
    x = pd.DataFrame([fila_1, fila_2])

    transformado = preprocesador.fit_transform(x)
    ordinales = transformado[:, -4:]
    assert ordinales[1, 0] > ordinales[0, 0]
