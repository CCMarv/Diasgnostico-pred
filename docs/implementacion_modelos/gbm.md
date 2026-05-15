# Gradient Boosting (GBM): implementación real en el proyecto

Resumen rápido:
- Clase usada: `GradientBoostingClassifier`.
- Definición en: `entrenamiento/comparador_modelos.py`.
- Parámetros clave: `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`, `random_state=SEMILLA_ALEATORIA`.

Idea central

GBM es un ensamblado de árboles secuenciales que corrige errores residuales de modelos previos. En este proyecto se usa como el candidato con mayor capacidad predictiva.

Equivalencias de código

Ejemplo didáctico (clase):
```python
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=100)
gbm.fit(X_train, y_train)
```

Implementación real (catálogo):
```python
GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=SEMILLA_ALEATORIA,
)
```

Diferencias importantes

- Parámetros ajustados para balance entre sesgo/varianza en datos clínicos.
- No se aplica `class_weight` (los estimadores de boosting manejan desbalance de forma distinta); si hace falta, el pipeline considera la recomendación del `CargadorDatos`.
- El preprocesador está presente en el pipeline pero GBM es menos sensible al escalado que SVM/MLP.

Entrenamiento y evaluación

- El pipeline se construye igual: `ConstructorPreprocesador().construir_pipeline(estimador)`.
- Se compara por `roc_auc` con cross-validation; si gana, se serializa.

Inferencia

Idéntico contrato: pipeline serializado cargado por `PredictorDiabetes`.

Recursos útiles

- Efecto del escalado: `docs/implementacion_modelos/ref/notas_escalar_o_no_escalar.md`.
- Explicabilidad: en `PROYECTO.md` se recomienda usar SHAP para GBM cuando se requiera justificabilidad clínica.

## Guía de implementación (fragmentos concretos)

Requisitos: `scikit-learn`, `pandas`, `joblib`, opcional `shap` para explicabilidad.

Ejecutar entrenamiento (pipeline) con GBM:

```bash
python -m entrenamiento.pipeline --modo clasificacion --modelos gbm
```

1) Ejemplo didáctico (clase)

```python
from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(n_estimators=100, max_depth=3)
gbm.fit(X_train, y_train)
```

2) Equivalente en el repo

```python
from entrenamiento.preprocesador import ConstructorPreprocesador
from sklearn.ensemble import GradientBoostingClassifier

estimador = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
pipeline = ConstructorPreprocesador().construir_pipeline(estimador)
pipeline.fit(X_train, y_train)
# serializar: joblib.dump(pipeline, 'modelos/gbm.joblib')
```

3) Explicabilidad (opcional)

```python
import shap
explainer = shap.Explainer(pipeline.named_steps['clasificador'], pipeline.named_steps['preprocesador'].transform(X_train))
shap_values = explainer(pipeline.named_steps['preprocesador'].transform(X_sample))
shap.summary_plot(shap_values, features=X_sample_transformed)
```

Notas:
- GBM maneja interacciones complejas; la normalización no es tan crítica como en SVM/MLP, pero mantener el pipeline garantiza reproducibilidad.
