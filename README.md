# Diasgnostico-pred

Primera espiral del proyecto de diagnóstico de riesgo de diabetes (CDC) con arquitectura modular en español.

## Estructura base

- `config.py`: constantes globales, rutas y columnas CDC.
- `entrenamiento/`: carga de datos, comparación de modelos y pipeline CLI.
- `entrenamiento/preprocesador.py`: `ColumnTransformer` serializable para evitar data leakage.
- `entrenamiento/evaluador.py`: métricas clínicas y generación de curvas/reportes.
- `inferencia/`: predictor para cargar artefactos `.joblib` y predecir.
- `api/`: esquemas Pydantic y API REST con FastAPI.
- `pruebas/`: pruebas de contrato iniciales.
- `notebooks/01_eda_regionalizado.ipynb`: guía de EDA regionalizado CDC ↔ ENSANUT.
- `reportes/contraste_regional.md`: contraste metodológico base para informe académico.

## Ejecución local

```bash
python -m pip install -e .[dev]
pytest
uvicorn api.main:app --reload
python -m entrenamiento.pipeline --modo clasificacion --dataset /ruta/al/dataset.csv
```

## Sprints (enfoque espiral)

1. Scaffolding y contratos base (este sprint).
2. Carga/limpieza y partición robusta del dataset.
3. Comparación de modelos y persistencia avanzada.
4. Inferencia y API productiva.
5. Observabilidad, endurecimiento y documentación final.
