from __future__ import annotations

import joblib
import pandas as pd

from config import COLUMNAS_CDC
from inferencia.predictor import PredictorDiabetes


class ModeloConProba:
    def predict_proba(self, x):
        return [[0.25, 0.75] for _ in range(len(x))]


class ModeloSinProba:
    def predict(self, x):
        return [1 for _ in range(len(x))]


def _entrada_valida() -> pd.DataFrame:
    return pd.DataFrame([{col: 1 for col in COLUMNAS_CDC}])


def test_predictor_con_predict_proba(tmp_path):
    ruta_modelo = tmp_path / "modelo.joblib"
    joblib.dump(ModeloConProba(), ruta_modelo)

    predictor = PredictorDiabetes(ruta_modelo=ruta_modelo)
    assert predictor.cargar_modelo() is True

    salida = predictor.predecir(_entrada_valida())
    assert salida["probabilidad"] == 0.75
    assert salida["clase"] == 1


def test_predictor_sin_predict_proba(tmp_path):
    ruta_modelo = tmp_path / "modelo.joblib"
    joblib.dump(ModeloSinProba(), ruta_modelo)

    predictor = PredictorDiabetes(ruta_modelo=ruta_modelo)
    predictor.cargar_modelo()

    salida = predictor.predecir(_entrada_valida())
    assert salida["probabilidad"] == 1.0
    assert salida["clase"] == 1
