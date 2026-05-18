from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from entrenamiento import comparador_modelos as modulo_comparador
from entrenamiento.comparador_modelos import ComparadorModelos


def test_evaluar_por_folds_usa_backend_loky(monkeypatch):
    parametros_parallel: dict[str, object] = {}
    tareas_capturadas: list[tuple] = []

    class ParallelFalso:
        def __init__(self, *args, **kwargs):
            parametros_parallel.update(kwargs)

        def __call__(self, _tareas):
            tareas_capturadas.extend(list(_tareas))
            return []

    monkeypatch.setattr(modulo_comparador, "Parallel", ParallelFalso)

    comparador = ComparadorModelos()
    x = pd.DataFrame({"x1": [0.0, 1.0, 0.5, 1.5]})
    y = pd.Series([0, 1, 0, 1])
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    puntajes = comparador._evaluar_por_folds(
        nombre="gbm",
        pipeline=object(),
        cv=cv,
        x_entrenamiento=x,
        y_entrenamiento=y,
        informar_progreso=None,
        etiqueta="gbm",
    )

    assert puntajes == []
    assert parametros_parallel.get("backend") == "loky"
    assert "prefer" not in parametros_parallel
    assert tareas_capturadas, "Se esperaban tareas de fold para ejecutar en paralelo."
    for tarea in tareas_capturadas:
        kwargs = tarea[2]
        assert kwargs["informar_progreso"] is None
