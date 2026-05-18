---
título: GBM — detalles específicos
categoría: referencia
audiencia: equipo técnico
versión: 1.0.0
última actualización: 2026-05-18
autor: Equipo diasgnostico-pred
estado: revisado
---

# GBM: Características y parámetros específicos

Esta guía complementa [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md). Lee primero la guía unificada para entender el flujo general (secciones 1-8), luego usa este documento para detalles específicos de Gradient Boosting.

## Para empezar: ¿qué hace este modelo en palabras simples?

### A. Definición coloquial

Imagina un equipo de correctores de exámenes: el primero revisa un examen y califica; el segundo revisa los errores del primero y los corrige; el tercero corrige los errores del segundo, y así 200 veces. Cada corrector es un árbol de decisión pequeño. Al final, la calificación del equipo completo es mucho más precisa que la de cualquier corrector individual. Eso es Gradient Boosting: 200 árboles pequeños que se construyen uno encima de los errores del anterior.

### B. Por qué lo usamos aquí

GBM es el modelo con mayor capacidad predictiva del catálogo del proyecto. Para detectar diabetes, donde los patrones son complejos y múltiples variables interactúan, este enfoque iterativo logra los mejores resultados en ROC-AUC.

### C. Qué significa que funcione bien o mal

- **Funciona bien**: el modelo generaliza correctamente a nuevos pacientes, con un ROC-AUC superior a 0.80, lo que significa alta discriminación entre casos positivos y negativos.
- **Funciona mal**: el modelo se entrena demasiado lento o sobreajusta cuando la tasa de aprendizaje es muy alta, memorizando el conjunto de entrenamiento sin aprender patrones reales.

### D. Glosario

| Término | Qué significa en lenguaje simple |
|---------|----------------------------------|
| Árbol débil | Un árbol muy pequeño (pocos nodos) que por sí solo no es muy preciso, pero contribuye poco a poco al conjunto |
| Tasa de aprendizaje | Qué tan grande es el paso de corrección en cada iteración; valores pequeños = más estable pero más lento |
| `n_estimators` | Cuántos árboles correctores se construyen en total; más árboles = más potente pero más lento |
| `max_depth` | Qué tan profundo puede ser cada árbol corrector; profundidad 4 es moderada |
| Gradiente | La dirección matemática en la que el modelo reduce sus errores en cada paso |
| Sobreajuste | Cuando el conjunto de 200 árboles aprende de memoria el entrenamiento y falla con datos nuevos |
| Validación cruzada | Entrenar y evaluar en distintas particiones de los datos para estimar el rendimiento real |

---

## Introducción: ¿Por qué GBM?

Gradient Boosting combina **árboles secuenciales**: cada nuevo árbol aprende a corregir los errores de los anteriores. En el proyecto es el modelo con **mayor capacidad predictiva** del catálogo supervisado.

**Ventajas:**
- Rendimiento típicamente superior a SVM y árbol simple
- Flexible, captura relaciones complejas
- No tan sensible al escalado como SVM

**Desventajas:**
- Menos interpretable que árbol simple
- Requiere SHAP para explicabilidad clínica
- Entrenamiento más lento

---

## Parámetros específicos de GBM

### Definición en el proyecto

```python
from sklearn.ensemble import GradientBoostingClassifier

estimador = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
)
```


### Justificación de cada parámetro (detallada)

- **`n_estimators=200`**
    - Qué controla: número total de árboles que se suman secuencialmente.
    - Por qué 200: ofrece capacidad suficiente sin multiplicar excesivamente el tiempo; con `learning_rate=0.05` es un buen trade-off entre potencia y costo.
    - Interacción: más estimators + menor learning_rate → mejor generalización pero mayor costo computacional.
    - Rango recomendado: probar [100, 200, 300]; para búsqueda fina combinar con `learning_rate`.

- **`max_depth=4`**
    - Qué controla: complejidad de cada árbol base (interacciones por árbol).
    - Por qué 4: árboles poco profundos evitan que cada estimador memorize ruido; el boosting compone muchas correcciones simples.
    - Rango recomendado: 3–6. Profundidades mayores aumentan riesgo de overfitting.

- **`learning_rate=0.05`**
    - Qué controla: cuánto aporta cada árbol nuevo a la predicción final.
    - Por qué 0.05: conservador y estable; reduce la varianza del ensamblado y evita saltos grandes.
    - Efecto práctico: si se reduce a 0.01, conviene aumentar `n_estimators`; si se sube a 0.1, entrenamiento más rápido pero mayor riesgo de sobreajuste.

- **`random_state=42`**
    - Reproducibilidad de inicializaciones y comportamientos internos.

- **Consideraciones adicionales**
    - `subsample` (muestreo por árbol): valores <1.0 introducen estocasticidad y pueden mejorar generalización (ej. 0.8).
    - `max_features`: limitar el número de características por árbol reduce correlación entre árboles.
    - Explicabilidad: usar SHAP para traducir decisión del ensamble a contribuciones por variable.

Resumen: la configuración prioriza generalización y estabilidad. Para ajustes, explore la relación `n_estimators × learning_rate` antes de cambiar profundidad de árboles.

### Comprensión intuitiva: n_estimators vs max_depth

- **Boosting = suma de correcciones:** Árbol 1 predice, Árbol 2 corrige parte del error del Árbol 1, Árbol 3 corrige parte del error del Árbol 2, etc.
- **n_estimators=200:** Suma de 200 correcciones pequeñas → muy expresivo
- **max_depth=4:** Cada árbol es simple → no memoriza, mantiene regularización
- **learning_rate=0.05:** Cada corrección es pequeña → convergencia estable

### Comparación con Árbol simple

| Aspecto | Árbol solo | GBM |
|---------|-----------|-----|
| **Operación** | Un corte de decisión | Suma de 200 correcciones |
| **Capacidad** | Baja/media | Alta |
| **Sobreajuste** | Alto sin límites | Controlado con learning_rate |

---

## Escalado NO obligatorio pero compatible

Resumen en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md): GBM (basado en árboles) no exige escalado, pero el `Pipeline` compartido puede aplicar `StandardScaler` sin perjudicar su comportamiento.

---

## Búsqueda de hiperparámetros (opcional)

El proyecto NO hace GridSearchCV por defecto en GBM. Se usa como uno de los candidatos fuertes del catálogo. Si quisieras optimizar:

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold

grid_params = {
    "clasificador__n_estimators": [100, 200, 300],
    "clasificador__learning_rate": [0.01, 0.05, 0.1],
    "clasificador__max_depth": [3, 4, 5],
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
```

**Nota:** Con 3×3×3=27 combinaciones × 5 pliegues = 135 entrenamientos. Es caro, por eso el proyecto usa valores por defecto.

---

## Explicabilidad: SHAP para GBM

A diferencia del Árbol, GBM no es legible directamente. Para contexto clínico auditable, el proyecto recomienda **SHAP** (ver [PROYECTO.md](../../PROYECTO.md)).

### Instalación

```bash
pip install shap
```

### Uso básico

```python
import shap

# Obtener el clasificador del Pipeline
clasificador = pipeline.named_steps["clasificador"]

# Crear explainer
explainer = shap.Explainer(clasificador, X_train_transformado)

# Calcular valores SHAP
shap_values = explainer(X_sample_transformado)

# Visualizaciones
shap.summary_plot(shap_values, X_sample_transformado, 
                  feature_names=nombres_columnas)

# Para una predicción individual
shap.waterfall_plot(shap_values[0])

# Dependencia de una variable
shap.dependence_plot("BMI", shap_values.values, X_sample_transformado)
```

### Interpretación SHAP

- **Summary plot:** "¿Qué variables son más importantes? ¿En qué dirección?"
- **Waterfall plot:** "¿Por qué el modelo predijo esta probabilidad para este paciente?"
- **Dependence plot:** "¿Cómo varía la predicción con esta variable?"

**Ejemplo de lectura:** "Para este paciente, Age=5 (+0.15 hacia riesgo) y MentHlth=30 (+0.08) son los mayores contribuyentes al riesgo predicho."

---

## Diferencias respecto a sklearn básico

| Aspecto | sklearn básico | Proyecto |
|---------|---------------|----------|
| **Parámetros** | Por defecto | Tuned (n_estimators=200, learning_rate=0.05) |
| **Desbalance** | Ignorado | Evaluado con ROC-AUC (métrica sensible a desbalance) |
| **Búsqueda** | No | No por defecto (valores fijos) |
| **Explicabilidad** | Importancia simple | SHAP si se necesita |

---

## Ventajas de GBM para el problema clínico

1. **Rendimiento:** Típicamente el mejor ROC-AUC del catálogo
2. **Flexibilidad:** Captura interacciones complejas (ej: Age × BMI × MentHlth)
3. **Estabilidad:** learning_rate bajo mantiene convergencia suave
4. **Interpretabilidad con SHAP:** Si bien no es simple como Árbol, SHAP permite explicaciones rigurosas

---
## Coherencia con el código y notas para no-programadores

- Implementación real: el catálogo en `entrenamiento/comparador_modelos.py` crea un `GradientBoostingClassifier` con `n_estimators=200`, `max_depth=4`, `learning_rate=0.05` y `random_state` fijo. En el pipeline se conserva este estimador por defecto (no hay Grid para GBM en el catálogo actual).

Explicación en lenguaje llano:

- "El modelo está configurado para sumar 200 árboles pequeños (cada uno con profundidad 4). Esto da mucha capacidad para aprender relaciones complejas, pero el parámetro `learning_rate=0.05` asegura que cada árbol aporte poco a la vez, evitando cambios bruscos y mejorando generalización."

Acción recomendada: los documentos de experimento deben anotar que GBM usa estos valores por defecto y que la búsqueda de parámetros es opcional y costosa.

Actualización del repositorio: el catálogo sigue usando estos hiperparámetros por defecto en `entrenamiento/comparador_modelos.py`; si se requiere una búsqueda formal, debe documentarse aparte para no mezclar el baseline con experimentos adicionales.

## Referencia de capas en el proyecto

1. **Definición:** [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py) - instancia GradientBoostingClassifier
2. **Preprocesamiento:** [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py) - ColumnTransformer
3. **Entrenamiento:** [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py) - cross_val_score con StratifiedKFold
4. **Serialización:** `modelos/predictor_gbm.joblib`
5. **Carga e inferencia:** [inferencia/predictor.py](../../inferencia/predictor.py)
6. **Exposición:** [api/main.py](../../api/main.py)

---


## Lectura complementaria

- Resúmenes clave integrados en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md) (escalado, desbalance, intuiciones).
- **Explicabilidad:** [PROYECTO.md](../../PROYECTO.md)
- **SHAP docs:** [GitHub SHAP](https://github.com/slundberg/shap)
- **Sklearn GradientBoosting:** [Documentación oficial](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)

---

## Volver a la guía unificada

Para entender el flujo completo: [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md)
