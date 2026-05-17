# Árbol de Decisión: Características y parámetros específicos

Esta guía complementa [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md). Lee primero la guía unificada para entender el flujo general (secciones 1-8), luego usa este documento para detalles específicos del Árbol.

## Introducción: ¿Por qué Árbol?

El Árbol de Decisión genera **reglas interpretables** del tipo "si X > umbral entonces…". En el proyecto actúa como modelo baseline explicable, útil para entender el problema y comparar con modelos más complejos.

**Ventajas:**
- Totalmente interpretable (puedes leer las reglas)
- No requiere escalado
- Sirve como explicación clínica

**Desventajas:**
- Propenso a sobreajuste si crece demasiado
- Menos flexible que GBM para relaciones complejas

---

## Parámetros específicos del Árbol

### Definición en el proyecto

```python
from sklearn.tree import DecisionTreeClassifier

estimador = DecisionTreeClassifier(
    max_depth=5,
    ccp_alpha=0.0,
    class_weight="balanced",
    random_state=42,
)
```


### Justificación de cada parámetro (detallada)

- **`max_depth=5`**
    - Qué controla: la profundidad máxima del árbol (número de decisiones encadenadas).
    - Por qué elegimos 5: limita la complejidad para mantener interpretabilidad y reducir varianza; con 21 variables una profundidad moderada genera reglas comprensibles (~hasta 31 nodos).
    - Rango recomendado: probar entre 3–8 durante análisis; profundidades >10 suelen sobreajustar en este dataset.
    - Efecto práctico: mayor profundidad → más capacidad (mejor fit en train), pero mayor riesgo de sobreajuste y reglas menos claras.

- **`ccp_alpha=0.0` (poda desactivada)**
    - Qué controla: fuerza de la poda por coste-complejidad (post-pruning).
    - Por qué lo dejamos en 0: para tener un baseline simple y comparables entre ejecuciones; la poda puede activarse si se observa sobreajuste consistente.
    - Recomendación: si el gap train/test es grande, explorar `ccp_alpha` mediante búsqueda (p. ej. valores pequeños 1e-4–1e-2) y validar la simplicidad vs. rendimiento.

- **`class_weight="balanced"`**
    - Qué controla: ajusta pesos inversamente proporcionales a la frecuencia de clases durante el entrenamiento.
    - Por qué: la clase positiva (~13%) es minoritaria; sin pesos el árbol tendería a clasificar la clase mayoritaria, degradando sensibilidad/recall.
    - Efecto práctico: mejora la detección de positivos a costa de posibles falsos positivos; balancea la métrica preferida (ROC-AUC/PR-AUC).

- **`random_state=42`**
    - Qué controla: semilla para operaciones aleatorias (split internos, búsqueda, etc.).
    - Por qué: reproducibilidad absoluta entre ejecuciones (útil para auditoría y comparaciones).

- **Parámetros adicionales a considerar**
    - `min_samples_split` / `min_samples_leaf`: pueden usarse para evitar particiones con muy pocos ejemplos; valores típicos 2–10 o fracciones (0.01–0.05).
    - `max_features`: limitar variables por corte puede reducir sobreajuste; útil cuando hay muchas entradas ruidosas.

En resumen, los parámetros priorizan interpretabilidad y robustez frente a overfitting; la estrategia práctica es mantener un baseline simple (`max_depth=5`, `class_weight="balanced"`) y ajustar únicamente si el análisis de gap train/test lo exige.

### Comprensión intuitiva: max_depth

- **max_depth=1:** Árbol trivial (solo 1 corte), muy sesgado pero sin varianza
- **max_depth=5:** Equilibrio → 31 nodos máximo, interpretable pero con capacidad
- **max_depth=100:** Árbol profundo, memoriza el entrenamiento, no generaliza

El proyecto elige 5 como punto de equilibrio.

---

## Escalado

- **SVM, MLP, K-Means:** escalado obligatorio (`StandardScaler`).
- **Árbol y GBM:** escalado no obligatorio (basados en umbrales).

Para el árbol específicamente:
- El árbol toma decisiones por **umbrales** (cortes), no por distancias; por eso **es insensible** a escalas absolutas.
- En producción el árbol comparte el `Pipeline` común, por lo que el escalado puede aparecer en el flujo sin cambiar la lógica del árbol.

---

## Interpretación del Árbol

### Extraer reglas en texto

```python
from sklearn.tree import export_text

# Una vez entrenado el pipeline
clasificador = pipeline.named_steps["clasificador"]

# Exportar como texto
reglas = export_text(clasificador, feature_names=nombres_columnas)
print(reglas)
```

**Ejemplo de salida:**
```
|--- BMI <= 25.00
|   |--- MentHlth <= 0.00
|   |   |--- class: 0
|   |--- MentHlth > 0.00
|   |   |--- class: 1
|--- BMI > 25.00
|   |--- Age <= 3.00
|   |   |--- class: 0
|   |--- Age > 3.00
|   |   |--- class: 1
```

**Lectura:** "Si BMI <= 25 Y MentHlth <= 0, entonces predice clase 0 (sin diabetes)"

### Visualización gráfica

```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(clasificador, 
          feature_names=nombres_columnas,
          class_names=["Sin diabetes", "Con diabetes"],
          filled=True,
          ax=ax)
plt.show()
```

---

## Diferencias respecto a sklearn básico

| Aspecto | sklearn básico | Proyecto |
|---------|---------------|----------|
| **Profundidad** | No especificada (puede crecer ilimitado) | max_depth=5 |
| **Desbalance** | Ignorado | Compensado con class_weight |
| **Búsqueda** | No | No (baseline simple) |
| **Validación** | Cross-val por defecto | StratifiedKFold en comparador |

---

## Ventajas clínicas de interpretabilidad

Diferencia principal respecto a SVM o MLP:

- **Médico/investigador** puede leer: "El árbol recomienza con desbalance cuando Age > X e Income > Y"
- **Auditoría regulatoria** más fácil: "Estas son exactamente las reglas que usa el sistema"
- **Explicación a paciente** más directa: "Se detectó riesgo porque combinas X + Y"

---

## Cuándo preferir Árbol

- **Principio de proyecto:** Entender la línea base interpretable
- **Debugging:** Verificar que características clínicas clave son usadas
- **Validación clínica:** Confirmar que reglas tienen sentido médico
- **Comparación:** Saber si modelos complejos realmente mejoran vs. baseline

---

## Coherencia con el código y notas para no-programadores

- Implementación real: `entrenamiento/comparador_modelos.py` crea el `DecisionTreeClassifier` con `max_depth=5`, `ccp_alpha=0.0`, `class_weight='balanced'` y `random_state` fijo. El árbol se entrena por defecto sin grid de búsqueda.

Explicación en lenguaje llano:

- "El árbol está configurado para ser relativamente pequeño (profundidad 5) para que las reglas que genera sean fáciles de leer. También ajusta pesos de clase para detectar mejor los casos de diabetes, que son menos frecuentes." 

Acción recomendada: documentar en experimentos si se cambia `max_depth` o se activa `ccp_alpha` para poda, y conservar `class_weight='balanced'` salvo justificación clínica explícita.

Actualización del repositorio: este documento sigue siendo coherente con `entrenamiento/comparador_modelos.py`; la lógica de búsqueda formal de hiperparámetros vive aparte en `entrenamiento/optimizador.py` y no altera el baseline del árbol.

## Referencia de capas en el proyecto

1. **Definición:** [entrenamiento/comparador_modelos.py](../../entrenamiento/comparador_modelos.py) - instancia DecisionTreeClassifier
2. **Preprocesamiento:** [entrenamiento/preprocesador.py](../../entrenamiento/preprocesador.py) - ColumnTransformer (escalado opcional)
3. **Entrenamiento:** [entrenamiento/pipeline.py](../../entrenamiento/pipeline.py) - cross_val_score con StratifiedKFold
4. **Serialización:** `modelos/modelo_diabetes_arbol.joblib`
5. **Carga e inferencia:** [inferencia/predictor.py](../../inferencia/predictor.py)
6. **Exposición:** [api/main.py](../../api/main.py)

---


## Lectura complementaria

- Resúmenes clave integrados en [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md) (escalado, manejo de desbalance, intuiciones de modelado).
- **Sklearn docs:** [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)

---

## Volver a la guía unificada

Para entender el flujo completo: [GUIA_UNIFICADA.md](../GUIA_UNIFICADA.md)
