---
título: SVM — detalles específicos
categoría: referencia
audiencia: equipo técnico
versión: 1.0.0
última actualización: 2026-05-18
autor: Equipo diasgnostico-pred
estado: revisado
---

# SVM: Características y parámetros específicos

Esta guía complementa [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md). Lee primero la guía unificada para entender el flujo general (secciones 1-8), luego usa este documento para detalles específicos de SVM.

## Para empezar: ¿qué hace este modelo en palabras simples?

### A. Definición coloquial

Imagina que tienes que separar manzanas de naranjas en una mesa. Una Máquina de Vectores de Soporte (SVM) busca la línea que deja la mayor distancia posible entre las manzanas más cercanas al borde y las naranjas más cercanas al borde. Cuanto más espacio entre ambos grupos, más segura es la separación. En este proyecto, en lugar de manzanas y naranjas, el modelo separa personas con riesgo de diabetes de personas sin riesgo, usando 21 características de salud como el IMC, la presión arterial y la edad.

### B. Por qué lo usamos aquí

La SVM es buena para encontrar patrones complejos cuando las variables de salud no se separan de forma simple o lineal. No basta con decir "si el IMC es alto, hay diabetes": la relación depende también de la edad, los antecedentes y otros factores al mismo tiempo.

### C. Qué significa que funcione bien o mal

- **Funciona bien**: de cada 10 personas con diabetes real, el modelo identifica 7 u 8 correctamente. Los médicos pueden priorizar a esas personas para revisiones adicionales.
- **Funciona mal**: el modelo clasifica como "sin riesgo" a personas que sí tienen diabetes. Eso puede retrasar un diagnóstico importante.

### D. Glosario

| Término | Qué significa en lenguaje simple |
|---------|----------------------------------|
| Kernel RBF | Una función matemática que permite al modelo encontrar separaciones curvas, no solo líneas rectas |
| Margen | El espacio vacío entre los dos grupos justo en la frontera de la separación |
| Parámetro C | Qué tan estricto es el modelo: valores altos = muy estricto, puede memorizar el entrenamiento |
| `class_weight='balanced'` | Le dice al modelo que preste igual atención a los casos de diabetes (pocos) que a los casos sin diabetes (muchos) |
| `predict_proba` | La capacidad del modelo de decir "creo que hay un 70% de probabilidad de diabetes" en lugar de solo "sí o no" |
| Validación cruzada | Entrenar y probar el modelo varias veces con distintas partes de los datos, para asegurarse de que los resultados son estables |
| ROC-AUC | Una puntuación de 0 a 1 que indica qué tan bien distingue el modelo entre casos con y sin diabetes; 0.5 es azar puro, 1.0 es perfecto |

---

## Introducción: ¿Por qué SVM?

La Máquina de Vectores de Soporte (SVM) busca el hiperplano que **maximiza el margen** entre clases. En el proyecto actúa como modelo baseline kernelizado, con capacidad para capturar fronteras no lineales a través de kernels (aquí: RBF).

**Ventajas:**
- Teoría bien establecida (margen, vectores de soporte)
- Escalable a dimensiones altas
- Control preciso a través de C y gamma

**Desventajas:**
- Requiere escalado obligatorio
- Búsqueda de hiperparámetros puede ser costosa
- Menos interpretable que un árbol

---

## Parámetros específicos de SVM

### Definición en el proyecto

```python
from sklearn.svm import SVC

estimador = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42,
)
```

### Justificación de cada parámetro

| Parámetro | Valor | Por qué |
|-----------|-------|--------|
| **kernel** | `"rbf"` | RBF (Radial Basis Function) permite fronteras no lineales sin diseñar interacciones a mano. Alternativa lineal sería para datos separables linealmente. |
| **C** | `1.0` | Controla cuánto penaliza el modelo los errores. 1.0 es un equilibrio: permite holgura en el margen pero penaliza cruces. La búsqueda explora [0.1, 1, 10]. |
| **gamma** | `"scale"` | Controla el alcance del kernel RBF. `"scale"` = 1/(n_features * X.var()), adapta la sensibilidad a la dispersión real de datos escalados. |
| **probability** | `True` | Permite obtener probabilidades (no solo clases). Necesario para consumo posterior en API y comparación con otros modelos. |
| **class_weight** | `"balanced"` | Compensa desbalance: clase positiva (diabetes) es minoritaria (~13%). Sin esto, el modelo se inclina hacia clase mayoritaria. |
| **random_state** | `42` | Reproducibilidad en todas las ejecuciones. |

### Comprensión intuitiva: Margen y C

Ver los resúmenes embebidos en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md) para la geometría de SVM. En breve:

- **Margen:** Distancia entre el hiperplano y los puntos más cercanos (vectores de soporte).
- **C pequeño (0.1):** Más holgura → margen más ancho, tolera errores.
- **C grande (10):** Más penalización → margen estrecho, mejor ajuste local.
- **C=1.0:** Equilibrio práctico para estos datos.

---

## Búsqueda de hiperparámetros para SVM

### Grid específico del proyecto

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

grid_params = {
    "clasificador__C": [0.1, 1, 10],
    "clasificador__gamma": ["scale", "auto"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

busqueda = GridSearchCV(
    estimator=pipeline,
    param_grid=grid_params,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)
busqueda.fit(X_train, y_train)
mejor_modelo = busqueda.best_estimator_
```

### Interpretación del grid

- **C=[0.1, 1, 10]:** Explora desde regularización fuerte hasta flexible
- **gamma=["scale", "auto"]:** Compara sensibilidad adaptada vs. por defecto
- **Total de combinaciones:** 6 (3 × 2), multiplicado por 5 pliegues = 30 entrenamientos

### Lectura rápida de hiperparámetros

Si solo necesitas lo más importante:

- **C:** "¿Cuánto castigo por error?" → Mayor C = penaliza más errores
- **gamma:** "¿Cuán local es la frontera?" → Mayor gamma = más sensible a vecinos cercanos
- **kernel="rbf":** "¿Curvatura de frontera?" → RBF permite cualquier curvatura
- **probability=True:** "Necesito probabilidades, no solo etiquetas"
- **class_weight="balanced":** "Hay desbalance, corrige automáticamente"

---

## Escalado obligatorio para SVM

Regla práctica (ver [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md) para más contexto): SVM requiere escalado; `StandardScaler` se ajusta solo en entrenamiento y se aplica por `Pipeline`.

Por qué: SVM usa distancias; variables con diferentes magnitudes sesgarían la solución sin escalado.

---

## Visualización 3D (referencia didáctica)

**Nota:** Esta sección es solo para entender la intuición geométrica. No forma parte del flujo de producción.

El notebook didáctico histórico visualiza un hiperplano sobre dos variables (conceptos integrados en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md)):

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Crear malla sobre el espacio de dos variables
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

# El hiperplano se obtiene desde coef_ e intercept_ (solo kernel lineal)
w = svm_model.coef_[0]
b = svm_model.intercept_[0]
zz = (-w[0] * xx - w[1] * yy - b)

# Visualizar
ax.plot_surface(xx, yy, zz, alpha=0.3, color="gray")
ax.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], 0, color="blue", label="Negativo")
ax.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], 0, color="red", label="Positivo")
ax.scatter(
    svm_model.support_vectors_[:, 0],
    svm_model.support_vectors_[:, 1],
    0,
    facecolors="none",
    edgecolors="black",
    s=100,
    label="Vectores de soporte"
)

ax.set_xlabel("Variable 1")
ax.set_ylabel("Variable 2")
ax.set_zlabel("Decisión")
ax.legend()
plt.show()
```

**¿Qué ves?**
- El **plano gris** es el hiperplano de decisión
- Los **puntos azules** son la clase negativa (sin diabetes)
- Los **puntos rojos** son la clase positiva (con diabetes)
- Los **puntos con círculos** son los vectores de soporte (puntos en el margen)

El margen es la distancia más pequeña entre el plano y los vectores de soporte.

---

## Diferencias respecto al notebook de referencia

| Aspecto | Notebook didáctico | Proyecto |
|---------|-------------------|----------|
| **Variables** | 2 (para visualización 3D) | 21 (CDC completo) |
| **Kernel** | `linear` | `rbf` |
| **Búsqueda** | No (valores fijos) | Sí (GridSearchCV) |
| **Validación** | Entrenamiento simple | StratifiedKFold |
| **Visualización** | 3D del hiperplano | No aplicable (21D) |

---

## Qué cambia en inferencia

El `Pipeline` entrenado se guarda con `joblib`:

```python
import joblib

joblib.dump(pipeline, "modelos/predictor_svm.joblib")
```

Al cargar y predecir:

```python
pipeline_cargado = joblib.load("modelos/predictor_svm.joblib")
X_nuevo = [...] # datos nuevos, NO escalados (el Pipeline lo hace)
prediccion = pipeline_cargado.predict(X_nuevo)
probabilidad = pipeline_cargado.predict_proba(X_nuevo)
```

El Pipeline automáticamente:
1. Aplica el preprocesamiento aprendido
2. Aplica la SVM entrenada
3. Retorna el resultado

---

## Coherencia con el código y notas para no-programadores

- Implementación real: en `entrenamiento/comparador_modelos.py` la SVM se define con `kernel='rbf'`, `class_weight='balanced'` y la búsqueda explora `C` en `[0.1, 1, 10]` y `gamma` en `['scale','auto']`.
- Proceso de búsqueda: durante la validación cruzada la SVM se instancia con `probability=False` (más rápido). Una vez elegida la mejor combinación, el código reentrena (`refit`) el estimador final con `probability=True` para que el modelo final guarde probabilidades útiles para la API.

Explicación en lenguaje llano:

- "Durante la fase de prueba probamos varias combinaciones sin pedir probabilidades para ahorrar tiempo; cuando elegimos la mejor combinación reentrenamos el modelo final y le pedimos que calcule probabilidades, que es lo que la API usa para dar un riesgo de diabetes." 

Acción recomendada: al documentar experimentos o al reproducir resultados, anotar que la búsqueda es rápida (sin probabilidades) y que el modelo final guarda probabilidades.

Actualización del repositorio: la búsqueda formal de hiperparámetros ahora queda encapsulada también en `entrenamiento/optimizador.py` mediante `OptimizadorHiperparametros`, con cobertura en `pruebas/test_optimizador.py`.


## Referencia de capas en el proyecto

1. **Definición:** [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py) - instancia SVC
2. **Preprocesamiento:** [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py) - ColumnTransformer + StandardScaler
3. **Entrenamiento:** [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py) - GridSearchCV con StratifiedKFold
4. **Serialización:** `modelos/modelo_diabetes_svm.joblib`
5. **Carga e inferencia:** [inferencia/predictor.py](../../inferencia/predictor.py)
6. **Exposición:** [api/main.py](../../api/main.py)

---

## Lectura complementaria

- Resúmenes clave integrados en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md) (escalado, desbalance, SVM-intuición).

---

## Volver a la guía unificada

Para entender el flujo completo: [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md)
