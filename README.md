# Diasgnostico-pred

Primera espiral del proyecto de diagnóstico de riesgo de diabetes (CDC) con arquitectura modular en español.

## Estructura base

- `config.py`: constantes globales, rutas y columnas CDC.
- `entrenamiento/`: carga de datos, comparación de modelos y pipeline CLI.
- `inferencia/`: predictor para cargar artefactos `.joblib` y predecir.
- `api/`: esquemas Pydantic y API REST con FastAPI.
- `pruebas/`: pruebas de contrato iniciales.

## Ejecución local

```bash
python -m pip install -e .[dev]
pytest
uvicorn api.main:app --reload
```

## Sprints (enfoque espiral)

1. Scaffolding y contratos base (este sprint).
2. Carga/limpieza y partición robusta del dataset.
3. Comparación de modelos y persistencia avanzada.
4. Inferencia y API productiva.
5. Observabilidad, endurecimiento y documentación final.
