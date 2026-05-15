# Árbol de Decisión: implementación real en el proyecto

Resumen rápido:
- Clase usada: `DecisionTreeClassifier`.
- Definición en: `entrenamiento/comparador_modelos.py`.
- Parámetros clave: `max_depth=5`, `ccp_alpha=0.0`, `class_weight='balanced'`, `random_state=SEMILLA_ALEATORIA`.

Idea central

Los árboles generan reglas de decisión (umbral por característica) que son naturalmente interpretable para clínicos. En este proyecto, el árbol se integra en el mismo `Pipeline` que el resto de modelos y se evalúa por `roc_auc` en validación cruzada.

Equivalencias de código

Ejemplo didáctico (clase):
```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
```

Implementación real (catálogo):
```python
DecisionTreeClassifier(
    max_depth=5,
    ccp_alpha=0.0,
    class_weight='balanced',
    random_state=SEMILLA_ALEATORIA,
)
```

Diferencias importantes

- `class_weight='balanced'` usado para compensar desbalance sin modificar filas (SMOTE/undersampling son alternativas que se documentan en `ref`).
- No hay `GridSearchCV` por defecto; el árbol actúa como baseline interpretable.
- Aunque los árboles no requieren `StandardScaler`, en el repo el árbol se envuelve en el `Pipeline` para mantener contrato unificado de columnas.

Entrenamiento y evaluación

- El pipeline se construye con `ConstructorPreprocesador().construir_pipeline(clasificador)`.
- Evaluación: `cross_val_score(pipeline, x, y, scoring='roc_auc')`.
- Si el árbol es el mejor, el pipeline completo se serializa con `joblib.dump`.

Inferencia

La API carga el pipeline serializado y llama a `predict_proba`/`predict` desde `inferencia/predictor.py`. El consumidor no distingue entre árbol o cualquier otro modelo.

Recursos útiles

- Escalado: `docs/implementacion_modelos/ref/notas_escalar_o_no_escalar.md` (los árboles no requieren escalado).
- Gestión del desbalance: `docs/implementacion_modelos/ref/Gestion_del_Desbalance_en_Machine_Learning.md`.

## Guía de implementación (fragmentos concretos)

Requisitos mínimos: `scikit-learn`, `pandas`, `joblib`.

Ejecutar entrenamiento (pipeline):

```bash
python -m entrenamiento.pipeline --modo clasificacion --modelos arbol
```

1) Ejemplo didáctico (clase)

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
```

2) Equivalente en el repo (pipeline integrado)

```python
from entrenamiento.preprocesador import ConstructorPreprocesador
from sklearn.tree import DecisionTreeClassifier

estimador = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
pipeline = ConstructorPreprocesador().construir_pipeline(estimador)
pipeline.fit(X_train, y_train)
# serializar con joblib.dump(pipeline, 'model.joblib')
```

3) Inferencia

```python
from inferencia.predictor import PredictorDiabetes
predictor = PredictorDiabetes(ruta_modelo=Path('modelos/modelo_diabetes_v1.joblib'))
predictor.cargar_modelo()
predictor.predecir(df_paciente)
```

Notas:
- Los árboles no requieren escalado, pero se mantienen en el `Pipeline` por consistencia del contrato de columnas.
- Usar `class_weight='balanced'` para casos con desbalance moderado en datos clínicos.
