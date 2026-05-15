# SVM: implementación real en el proyecto

Este archivo explica dónde vive la SVM dentro de `Diasgnostico-pred` y cómo se diferencia de la implementación típica de clase.

## Idea central

En un notebook de clase, la SVM suele aparecer como una secuencia lineal:

1. cargar datos,
2. seleccionar pocas variables,
3. escalar,
4. entrenar `SVC`,
5. graficar o evaluar.

En este proyecto la SVM no está aislada. Se integra dentro de un `sklearn.Pipeline` común, comparte preprocesamiento con otros modelos, se evalúa por validación cruzada y luego se serializa para que la API la consuma sin saber qué modelo concreto hay detrás.

## Equivalencias de código

### 1. Carga de datos

En un notebook de ejemplo normalmente se ve algo como:

```python
df = pd.read_csv('fraude.csv')
X = df[['monto', 'antiguedad_cuenta']].values
y = df['es_fraude'].values
```

En este repositorio, la entrada real del sistema no usa fraude ni dos variables. Usa el dataset CDC BRFSS 2015 y 21 columnas clínicas.

La definición de contrato de datos está en [config.py](../../config.py).

- `COLUMNAS_CDC` define las 21 variables de entrada.
- `COLUMNA_OBJETIVO` define la salida `Diabetes_binary`.

Eso significa que la SVM aquí trabaja con un problema clínico de diabetes, no con una visualización bidimensional.

### 2. Definición del modelo SVM

En el notebook de clase suele verse algo como:

```python
svm_model = SVC(kernel='linear', C=1.0)
```

En el proyecto, la SVM real se define en [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py) dentro del catálogo de modelos:

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=SEMILLA_ALEATORIA,
)
```

Diferencias importantes:

- `kernel="rbf"` en lugar de `linear`.
- `probability=True` para poder producir probabilidades y no solo clases.
- `class_weight="balanced"` para compensar desbalance.
- `random_state` para reproducibilidad.
- Se acompaña con búsqueda de hiperparámetros (`C` y `gamma`) por `GridSearchCV`.

### 3. Preprocesamiento

En clase el escalado suele hacerse así:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Aquí no se escala todo igual. El preprocesamiento está en [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py) y separa las columnas en tres grupos:

- continuas: imputación + `StandardScaler`,
- binarias: imputación + paso directo,
- ordinales: imputación + `OrdinalEncoder`.

La SVM no recibe `X` ya transformado manualmente. Recibe un `Pipeline` completo:

```python
pipeline = self._preprocesador.construir_pipeline(estimador)
```

Eso evita `data leakage` porque el preprocesamiento queda dentro del flujo de entrenamiento y validación.

### 4. Entrenamiento

En un notebook de clase normalmente se entrena una sola vez:

```python
svm_model.fit(X_scaled, y)
```

En el proyecto, la SVM se entrena dentro de [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py) mediante validación cruzada estratificada:

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA_ALEATORIA)
```

Y luego:

- si hay `grid`, se usa `GridSearchCV`,
- si no hay `grid`, se usa `cross_val_score`.

Para SVM sí existe búsqueda de hiperparámetros:

```python
"clasificador__C": [0.1, 1, 10]
"clasificador__gamma": ["scale", "auto"]
```

Aquí el prefijo `clasificador__` aparece porque la SVM está dentro de un `Pipeline` y ese es el nombre del paso final.

### 5. Evaluación

En clase muchas veces se mira una métrica simple o una gráfica manual.

En el proyecto, el entrenamiento general se orquesta en [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py):

- separa entrenamiento/prueba,
- entrena todos los modelos candidatos,
- calcula ROC-AUC sobre la prueba,
- selecciona el mejor,
- guarda métricas y artefactos.

La SVM compite con árbol, GBM y MLP bajo la misma métrica, no en un escenario aislado.

### 6. Serialización

En clase el modelo suele quedar en memoria.

Aquí el mejor pipeline se guarda con `joblib` en [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py):

```python
joblib.dump(mejor_resultado.modelo, ruta_modelo)
```

Eso significa que se serializa el pipeline completo, no solo el clasificador. Por eso luego la inferencia puede cargar un único archivo y usarlo directamente.

### 7. Inferencia

En un notebook de clase normalmente la predicción se hace sobre la misma sesión.

Aquí la carga y uso real están en [inferencia/predictor.py](../../inferencia/predictor.py):

```python
self._modelo = joblib.load(self.ruta_modelo)
```

Y después:

```python
if hasattr(self._modelo, "predict_proba"):
    proba_raw = self._modelo.predict_proba(entrada_ordenada)
```

La API no sabe si hay una SVM, un árbol o un GBM. Solo consume un predictor cargado desde disco.

## Flujo completo de SVM en el proyecto

1. La SVM se registra en el catálogo de [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py).
2. El preprocesamiento se construye en [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py).
3. El entrenamiento y la comparación de modelos se ejecutan en [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py).
4. El mejor pipeline se serializa con `joblib`.
5. [inferencia/predictor.py](../../inferencia/predictor.py) carga el archivo `.joblib`.
6. [api/main.py](../../api/main.py) expone `/predecir` y traduce la probabilidad en una categoría de riesgo.

## Qué cambia respecto a clase

- No hay una sola `SVC` independiente: la SVM vive dentro de un `Pipeline`.
- No se usan dos variables para graficar el hiperplano: se usan 21 variables clínicas.
- No se entrena sobre todo el dataset sin control: se usa validación cruzada y separación train/test.
- No se reporta solo una frontera de decisión: se guarda el modelo final para producción.
- La SVM no se interpreta como demo aislada, sino como un candidato dentro de un sistema comparativo.

## Qué mirar si quieres seguir el código

- Parámetros SVM: [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py)
- Preprocesamiento: [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py)
- Orquestación de entrenamiento: [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py)
- Contrato de inferencia: [inferencia/predictor.py](../../inferencia/predictor.py)
- Consumo en API: [api/main.py](../../api/main.py)

## Plantilla para futuros modelos

Cuando agreguemos Árbol, GBM, MLP o K-Means, cada archivo debería repetir esta misma estructura:

- idea central,
- equivalencias de código,
- definición real del modelo,
- preprocesamiento asociado,
- entrenamiento,
- evaluación,
- serialización,
- inferencia,
- diferencias respecto a clase.

## Recursos de referencia (carpeta `ref`)

He integrado y resumido aquí los recursos de la carpeta `docs/implementacion_modelos/ref` que resultan útiles para entender por qué el proyecto aplica ciertas decisiones (escalado, manejo de desbalance, visualizaciones didácticas).

- `notas_svm_fraude.md`: Notebook y notas que muestran una SVM lineal aplicada a un problema de fraude con 2 features y visualización 3D del hiperplano. Útil para la intuición geométrica de `coef_`, `intercept_` y `support_vectors_`. Ver [docs/implementacion_modelos/ref/notas_svm_fraude.md](ref/notas_svm_fraude.md).
- `notas_escalar_o_no_escalar.md`: Explica cuándo y por qué escalar; incluye ejemplos prácticos con `MinMaxScaler`/`StandardScaler` y la regla crítica: `fit` solo en train para evitar data leakage. Recomendación: escalar para SVM. Ver [docs/implementacion_modelos/ref/notas_escalar_o_no_escalar.md](ref/notas_escalar_o_no_escalar.md).
- `Gestion_del_Desbalance_en_Machine_Learning.md`: Resumen exhaustivo de métricas (ROC vs PR), técnicas de re-balanceo (SMOTE, NearMiss) y ejemplos con `imblearn`. Explica por qué el proyecto usa `class_weight='balanced'` y cuándo preferir SMOTE/undersampling. Ver [docs/implementacion_modelos/ref/Gestion_del_Desbalance_en_Machine_Learning.md](ref/Gestion_del_Desbalance_en_Machine_Learning.md).
- `smv_fraude.ipynb` y `gestion_desbalance.ipynb`: notebooks de apoyo con código reproducible y experimentos que ilustran los puntos anteriores.

Si quieres, puedo extraer fragmentos de código concretos (por ejemplo, la sección de escalado correcta o el ejemplo de SMOTE) y pegarlos aquí como comparaciones línea a línea contra las llamadas reales en `entrenamiento/preprocesador.py` y `entrenamiento/comparador_modelos.py`.

## Guía de implementación (fragmentos concretos)

Requisitos mínimos:

- Python ≥ 3.11 (ver `pyproject.toml`)
- Dependencias: `scikit-learn`, `pandas`, `joblib`, `matplotlib` (instalar con `pip install -e .[dev]`).

Cómo ejecutar el pipeline de entrenamiento (clasificación):

```bash
python -m entrenamiento.pipeline --modo clasificacion
```

1) Ejemplo didáctico (notebook): escalado + SVM lineal

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Preparación (2 features) -- demo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_scaled, y)
```

2) Equivalente en el repo: pipeline serializable + grid search

```python
from entrenamiento.preprocesador import ConstructorPreprocesador
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold

estimador = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
pre = ConstructorPreprocesador()
pipeline = pre.construir_pipeline(estimador)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = {'clasificador__C': [0.1, 1, 10], 'clasificador__gamma': ['scale', 'auto']}
busqueda = GridSearchCV(pipeline, param_grid=grid, cv=cv, scoring='roc_auc', n_jobs=-1)
busqueda.fit(X_train, y_train)
# Mejor pipeline serializable: busqueda.best_estimator_
```

3) Cómo cargar el modelo en producción (inferencer)

```python
from inferencia.predictor import PredictorDiabetes
predictor = PredictorDiabetes(ruta_modelo=RUTA_MODELO_FINAL)
predictor.cargar_modelo()
predictor.predecir(df_paciente)  # df_paciente: DataFrame con las 21 columnas CDC
```

Notas rápidas:
- Asegúrate de que el `StandardScaler` solo se ajuste en el `Pipeline` (ya lo hace `ConstructorPreprocesador`).
- Para experimentos locales rápidos puedes usar el notebook `docs/implementacion_modelos/ref/smv_fraude.ipynb`.
