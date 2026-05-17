import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from entrenamiento.optimizador import OptimizadorHiperparametros


def test_buscar_returns_gridsearchcv():
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=0)
    pipeline = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(solver="liblinear"))])
    param_grid = {"clf__C": [0.1, 1.0]}
    opt = OptimizadorHiperparametros(cv_splits=3, scoring="roc_auc", n_jobs=1)
    search = opt.buscar(pipeline, param_grid, X, y)
    # check that a search object with best_params_ exists
    assert hasattr(search, "best_params_")


def test_mejor_pipeline_has_estimator():
    X, y = make_classification(n_samples=150, n_features=6, n_informative=3, random_state=1)
    pipeline = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(solver="liblinear"))])
    param_grid = {"clf__C": [0.01, 0.1]}
    opt = OptimizadorHiperparametros(cv_splits=3, scoring="roc_auc", n_jobs=1)
    opt.buscar(pipeline, param_grid, X, y)
    best = opt.mejor_pipeline()
    assert hasattr(best, "predict")
