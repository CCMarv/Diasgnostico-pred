# MLP (Red neuronal multicapa): implementación real en el proyecto

Resumen rápido:
- Clase usada: `MLPClassifier`.
- Definición en: `entrenamiento/comparador_modelos.py`.
- Parámetros clave: `hidden_layer_sizes=(64,32)`, `activation='relu'`, `solver='adam'`, `max_iter=500`, `early_stopping=True`, `validation_fraction=0.1`, `random_state=SEMILLA_ALEATORIA`.

Idea central

Las redes MLP capturan interacciones no lineales complejas entre variables clínicas. En el proyecto se usan como uno de los candidatos con ajuste fino para evitar sobreajuste (early stopping).

Equivalencias de código

Ejemplo didáctico (clase):
```python
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200)
mlp.fit(X_train_scaled, y_train)
```

Implementación real (catálogo):
```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=SEMILLA_ALEATORIA,
)
```

Diferencias importantes

- **Escalado obligatorio**: las redes convergen mejor con `StandardScaler` (ver `preprocesador`).
- `early_stopping=True` evita iteraciones innecesarias y reduce overfitting.
- Validación interna (10%) usada por `MLPClassifier` para detener entrenamiento si no mejora.

Entrenamiento y evaluación

- Formato idéntico: pipeline + cross-validation por `roc_auc`.
- Si MLP gana, el pipeline completo se serializa.

Inferencia

- Predictor carga pipeline y llama a `predict_proba` si existe.

Recursos útiles

- Escalado y reglas de `fit` solo en train: `docs/implementacion_modelos/ref/notas_escalar_o_no_escalar.md`.
- Explicabilidad y comparación con GBM: `PROYECTO.md`.

## Guía de implementación (fragmentos concretos)

Requisitos: `scikit-learn`, `pandas`, `joblib`.

Ejecutar entrenamiento MLP:

```bash
python -m entrenamiento.pipeline --modo clasificacion --modelos mlp
```

1) Ejemplo didáctico (clase)

```python
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200)
mlp.fit(X_train_s, y_train)
```

2) Equivalente en el repo

```python
from entrenamiento.preprocesador import ConstructorPreprocesador
from sklearn.neural_network import MLPClassifier

estimador = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', early_stopping=True, max_iter=500, random_state=42)
pipeline = ConstructorPreprocesador().construir_pipeline(estimador)
pipeline.fit(X_train, y_train)
# serializar: joblib.dump(pipeline, 'modelos/mlp.joblib')
```

Notas:
- `StandardScaler` es crítico; en este repo está incorporado en `ConstructorPreprocesador`.
- `early_stopping=True` reduce riesgo de overfitting y acelera el desarrollo.
