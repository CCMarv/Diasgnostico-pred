# K-Means: uso como fenotipado (clustering) en el proyecto

Resumen rápido:
- Clase usada: `KMeans`.
- Definición en: `entrenamiento/comparador_modelos.py`.
- Parámetros clave: `n_clusters=3`, `random_state=SEMILLA_ALEATORIA`, `n_init='auto'`.

Idea central

K-Means se usa para fenotipado metabólico (descubrir subgrupos clínicamente interpretables), no como un predictor supervisado de diabetes. Se entrena sin la variable objetivo para evitar sesgo de confirmación.

Equivalencias de código

Ejemplo didáctico (clase):
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)
```

Implementación real (catálogo):
```python
KMeans(n_clusters=n_clusters, random_state=SEMILLA_ALEATORIA, n_init='auto')
```

Diferencias importantes

- Se entrena sobre `X` limpio (sin `Diabetes_binary`) y se evalúa por inercia / silhouette.
- La etiqueta de fenotipo puede añadirse más tarde como feature ordinal si se desea (`construir_pipeline_con_fenotipo`).

Entrenamiento y evaluación

- En `entrenamiento/pipeline.py` el flujo `modo='clustering'` llama a `ComparadorModelos.entrenar_clustering`.
- Se serializa el modelo final si es requerido.

Inferencia y uso clínico

- Los fenotipos ayudan a explicar subtipos de riesgo y a guiar análisis de equidad.
- No forman parte del pipeline supervisado a menos que explícitamente se use `construir_pipeline_con_fenotipo`.

Recursos útiles

- Notas sobre clustering y elección de K en `docs/implementacion_modelos/ref/*`.

## Guía de implementación (fragmentos concretos)

Requisitos: `scikit-learn`, `pandas`, `joblib`.

Ejecutar clustering:

```bash
python -m entrenamiento.pipeline --modo clustering --clusters 3
```

1) Ejemplo didáctico (clase)

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
```

2) Equivalente en el repo

```python
from entrenamiento.comparador_modelos import ComparadorModelos
comparador = ComparadorModelos()
mejor = comparador.entrenar_clustering(X_limpio.to_numpy(), n_clusters=3)
# mejor.modelo es el KMeans entrenado
```

3) Uso del fenotipo como feature (opcional)

```python
# Si deseas añadir fenotipo como columna ordinal
pipeline = ConstructorPreprocesador().construir_pipeline_con_fenotipo(clasificador)
```

Notas:
- Entrena sin la variable objetivo; evalúa con inercia o silhouette.
- No se añade automáticamente al pipeline supervisado a menos que se construya con `construir_pipeline_con_fenotipo`.
