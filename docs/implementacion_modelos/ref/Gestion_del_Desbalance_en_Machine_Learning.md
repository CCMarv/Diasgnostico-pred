# Gestión del Desbalance en Machine Learning

**Machine Learning — Procesamiento Avanzado de Datos**  
Licenciatura en Inteligencia Artificial  
Unidad: Procesamiento Avanzado de Datos

---

## Temario

| # | Tema | Descripción |
|---|------|-------------|
| 01 | El Problema del Desbalance | Fundamentos, proporciones típicas y consecuencias en ML |
| 02 | Métricas en Datos Desbalanceados | Precision-Recall vs ROC, cuándo usar cada métrica |
| 03 | SMOTE: Oversampling Sintético | Algoritmo, implementación y variantes |
| 04 | NearMiss: Undersampling Inteligente | Tres versiones del algoritmo y aplicaciones |
| 05 | ColumnTransformer en sklearn | Preprocesamiento avanzado para datos heterogéneos y pipelines integrados |

---

## Capítulo 01 — El Problema del Desbalance

### ¿Qué es el Desbalance de Clases?

Un dataset está desbalanceado cuando las categorías de clasificación no están representadas aproximadamente en igual proporción. La clase minoritaria (positiva) tiene significativamente menos ejemplos que la clase mayoritaria (negativa).

**¿Por qué es problemático?**

- Los algoritmos optimizan accuracy, ignorando la clase minoritaria.
- El modelo aprende patrones sesgados hacia la clase mayoritaria.
- Alta accuracy puede ocultar pobre rendimiento en la clase de interés.

### Proporciones Típicas

| Dominio | Ratio |
|---------|-------|
| Detección de Fraude | 1 : 1000 |
| Diagnóstico Médico | 1 : 100 |
| Detección de Spam | 1 : 10 |

> **Ejemplo ilustrativo:** En un dataset con 9,900 ejemplos negativos y 100 positivos, un clasificador que siempre predice "negativo" obtiene **99% de accuracy**, pero **0% de recall** en la clase positiva.

---

### Ejemplos Reales de Desbalance

| Dominio | Ratio | Consecuencia del falso negativo | Dataset |
|---------|-------|---------------------------------|---------|
| Fraude Bancario | 1 : 1,000 | Pérdida financiera significativa | Credit Card Fraud |
| Diagnóstico Médico | 1 : 100 | Diagnóstico tardío potencialmente fatal | Mammography |
| Ciberseguridad | 1 : 10,000 | Intrusiones no detectadas en redes | KDD Cup 1999 |
| Detección de Spam | 1 : 10 | Emails importantes en carpeta de spam | SpamAssassin |
| Mantenimiento Predictivo | 1 : 50 | Costosos paros no anticipados | NASA Bearing |
| Churn de Clientes | 1 : 5 | Pérdida de clientes sin intervención | Telco Customer |

---

## Capítulo 02 — Métricas en Datos Desbalanceados

### La Matriz de Confusión y Métricas Básicas

|  | Predicho Negativo | Predicho Positivo |
|--|-------------------|-------------------|
| **Real Negativo** | TN — Verdadero Negativo | FP — Falso Positivo |
| **Real Positivo** | FN — Falso Negativo | TP — Verdadero Positivo |

**El problema del Accuracy:**

```
Accuracy = (TP + TN) / Total
```

En un dataset 99:1, un clasificador "siempre negativo" obtiene 99% de accuracy. El accuracy es engañoso en datos desbalanceados.

---

### Métricas Clave

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **Precision** | `TP / (TP + FP)` | De todas las predicciones positivas, ¿cuántas son realmente positivas? Alta precision = pocos falsos alarmas. |
| **Recall** | `TP / (TP + FN)` | De todos los ejemplos realmente positivos, ¿cuántos detectamos? Alto recall = pocos positivos perdidos. |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Media armónica que balancea precision y recall. |

---

### Curvas ROC y AUC

La curva ROC evalúa el trade-off entre **True Positive Rate (TPR)** y **False Positive Rate (FPR)** a diferentes umbrales de decisión.

- **Eje Y (TPR):** `TP / (TP + FN)` — Sensibilidad / Recall
- **Eje X (FPR):** `FP / (FP + TN)` — 1 - Especificidad

**Interpretación del AUC:**

- AUC = 0.5 → Modelo equivalente al azar (línea diagonal)
- AUC = 1.0 → Clasificador perfecto
- Interpretación: Probabilidad de que el modelo rankee un ejemplo positivo aleatorio más alto que uno negativo aleatorio.

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calcular curva ROC
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Visualizar
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Azar')
```

---

### Curvas Precision-Recall y Average Precision

La curva PR muestra el trade-off entre **Precision** y **Recall** a diferentes umbrales.

- **Eje Y:** `Precision = TP / (TP + FP)`
- **Eje X:** `Recall = TP / (TP + FN)`

**Average Precision (AP):**

```
AP = Σ(Rₙ - Rₙ₋₁) × Pₙ
```

Donde Pₙ y Rₙ son precision y recall en el n-ésimo umbral.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Calcular curva PR
precision, recall, _ = precision_recall_curve(y_true, y_scores)
ap_score = average_precision_score(y_true, y_scores)

# Visualizar
plt.plot(recall, precision, label=f'PR (AP = {ap_score:.2f})')
plt.axhline(y=baseline, color='r', linestyle='--', label='Baseline')
```

---

### ROC vs Precision-Recall: Comparación Práctica

| Usar **ROC (AUC-ROC)** cuando… | Usar **Precision-Recall** cuando… |
|-------------------------------|----------------------------------|
| Clases balanceadas (proporción ~1:1) | Clase positiva rara (desbalance 1:100 o mayor) |
| FP y FN tienen costos comparables | Falsos positivos tienen alto costo |
| El rendimiento en ambas clases es relevante | Solo importa detectar la clase positiva |
| Se evalúa capacidad de discriminación general | Información retrieval: búsqueda, recomendación, detección de anomalías |

**Limitaciones de ROC:**
- FPR engañoso en datasets muy desbalanceados: puede parecer bajo incluso con muchos FP absolutos.
- Incluye TN en el cálculo, abundantes en la clase mayoritaria.
- Evalúa ambas clases por igual, sin énfasis en la clase de interés.

**Ventajas de PR:**
- No considera verdaderos negativos; se enfoca solo en predicciones positivas.
- Baseline claro: la línea base es la proporción de clase positiva.
- Cambios en PR son más visibles y significativos en datos desbalanceados.

> **Regla práctica:** Si la clase positiva representa **< 10%** del dataset → usa **Precision-Recall**. Si está entre **20–50%** → **ROC es adecuado**.

**Ejemplos:**

| ROC | Precision-Recall |
|-----|-----------------|
| Clasificación de imágenes (gato vs. perro) | Detección de fraude (1:1000) |
| Predicción de clima (lluvia vs. no lluvia) | Diagnóstico de cáncer (1:100) |
| Diagnóstico común (gripe vs. resfriado) | Detección de intrusiones (1:10000) |

---

## Capítulo 03 — SMOTE

### Fundamentos de SMOTE

**SMOTE** (Synthetic Minority Over-sampling Technique) genera ejemplos sintéticos de la clase minoritaria en lugar de simplemente replicar los existentes. Esto expande las regiones de decisión y evita el sobreajuste.

| Método | Resultado |
|--------|-----------|
| Replicación simple | Duplica puntos existentes → Overfitting |
| SMOTE | Interpola nuevos puntos → Generalización |

**Fundamento Matemático:**

Para cada muestra minoritaria **xᵢ**, SMOTE:
1. Encuentra sus **k vecinos más cercanos** en la clase minoritaria.
2. Selecciona aleatoriamente uno de estos vecinos **xₙₙ**.
3. Genera una muestra sintética interpolando:

```
x_new = xᵢ + λ × (xₙₙ - xᵢ)
```

donde `λ ~ U(0,1)` es un número aleatorio entre 0 y 1.

**Beneficios de SMOTE:**
- Mayor generalización: regiones de decisión más amplias y suaves.
- Menos overfitting: no replica datos, crea nuevos ejemplos.
- Mejor cobertura: expande el espacio de características de la clase minoritaria.
- Árboles más simples: menos nodos terminales en clasificadores de árbol.

---

### Algoritmo SMOTE Paso a Paso

**Pseudo-código:**

```
Input:  Minority samples T, N%, k
Output: (N/100) × T synthetic samples

1. if N < 100:
     Randomizar T, T = (N/100) × T
     N = 100
2. N = int(N/100)  → múltiplos de 100
3. for each xᵢ in T:
     Compute k nearest neighbors
     Save indices in nnarray
4. Populate(N, i, nnarray):
     while N ≠ 0:
       Choose random neighbor nn
       for each feature:
         diff = Sample[nn][attr] - Sample[i][attr]
         gap  = random(0, 1)
         Synthetic[attr] = Sample[i][attr] + gap × diff
       N = N - 1
```

**Ejemplo Numérico:**

| Parámetro | Valor |
|-----------|-------|
| Muestra minoritaria xᵢ | (6, 4) |
| Vecino seleccionado xₙₙ | (4, 3) |
| λ (aleatorio) | 0.5 |

```
x_new[0] = 6 + 0.5 × (4 - 6) = 5
x_new[1] = 4 + 0.5 × (3 - 4) = 3.5
→ Muestra sintética: (5, 3.5)
```

**Hiperparámetros Clave:**

| Parámetro | Descripción |
|-----------|-------------|
| `k` | Número de vecinos. Default: 5. |
| `N` | Porcentaje de oversampling (100%, 200%, etc.). |

---

### SMOTE en Python con imbalanced-learn

```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter

# 1. Crear dataset desbalanceado
X, y = make_classification(
    n_classes=2, class_sep=2,
    weights=[0.9, 0.1],       # 90% - 10%
    n_informative=3, n_redundant=1,
    n_features=20, n_samples=1000,
    random_state=42
)
print(f"Original: {Counter(y)}")
# Output: Counter({0: 900, 1: 100})

# 2. Aplicar SMOTE
smote = SMOTE(
    sampling_strategy=0.5,    # ratio 1:2
    k_neighbors=5,
    random_state=42
)
X_res, y_res = smote.fit_resample(X, y)
print(f"SMOTE: {Counter(y_res)}")
# Output: Counter({0: 900, 1: 450})
```

**Parámetros de SMOTE:**

| Parámetro | Descripción |
|-----------|-------------|
| `sampling_strategy` | `'auto'`, float o dict. Ratio deseado minoría:mayoría. |
| `k_neighbors` | Número de vecinos a considerar (default: 5). |
| `random_state` | Semilla para reproducibilidad. |

**Estrategias de sampling:**
- `'auto'` → Igualar número de muestras en ambas clases.
- `float` (ej. `0.5`) → Ratio minoría/mayoría = 0.5 (1:2).
- `dict` `{0: 900, 1: 450}` → Especificar número exacto por clase.

```bash
pip install imbalanced-learn
```

---

## Capítulo 04 — NearMiss

### Fundamentos de NearMiss

**NearMiss** es un algoritmo de undersampling inteligente que selecciona muestras de la clase mayoritaria basándose en su **distancia a la clase minoritaria**, en lugar de eliminar aleatoriamente.

| Método | Resultado |
|--------|-----------|
| Random Undersampling | Elimina muestras al azar → puede perder información valiosa |
| NearMiss | Selecciona muestras relevantes → preserva información de frontera |

**Proceso General:**

1. **Calcular Distancias** — Para cada muestra mayoritaria, calcular distancia a todas las muestras minoritarias.
2. **Identificar Vecinos** — Para cada muestra, encontrar los k vecinos minoritarios más cercanos (o lejanos).
3. **Seleccionar Muestras** — Aplicar heurística específica para elegir qué muestras mayoritarias conservar.

**Medida de distancia (Euclidiana):**

```
d(x, y) = √Σ(xᵢ - yᵢ)²
```

Para datasets grandes se pueden usar k-d trees o ball trees para mayor eficiencia.

---

### Las Tres Versiones de NearMiss

#### NearMiss-1

**Heurística:** Selecciona muestras mayoritarias cuya distancia promedio a los **k vecinos minoritarios más cercanos** sea la menor.

```
argmin (1/k × Σ d(x_major, x_minor_nn))
```

- Conserva muestras cercanas a minoría.
- Sensible a ruido en la clase minoritaria.
- Frontera de decisión más definida.

#### NearMiss-2

**Heurística:** Selecciona muestras mayoritarias cuya distancia promedio a los **k vecinos minoritarios más lejanos** sea la menor.

```
argmin (1/k × Σ d(x_major, x_minor_far))
```

- Considera la distribución global de la minoría.
- Menos sensible a ruido local.
- Mejor para minorías dispersas.

#### NearMiss-3

**Heurística (2 fases):**
- **Fase 1:** Para cada minoritario, encontrar *m* vecinos mayoritarios más cercanos.
- **Fase 2:** De ese conjunto, seleccionar los que tengan mayor distancia promedio a sus vecinos minoritarios.

```
argmax (1/k × Σ d(x_selected, x_minor_nn))
```

- Más robusto al ruido.
- Mayor margen de separación alrededor de la clase minoritaria.

---

### NearMiss en Python: Implementación Práctica

```python
from imblearn.under_sampling import NearMiss
from sklearn.datasets import make_classification
from collections import Counter

# Crear dataset desbalanceado
X, y = make_classification(
    n_classes=2, weights=[0.9, 0.1],
    n_samples=1000, random_state=42
)
print(f"Original: {Counter(y)}")
# Counter({0: 900, 1: 100})

# NearMiss-1
nm1 = NearMiss(version=1, n_neighbors=3)
X_nm1, y_nm1 = nm1.fit_resample(X, y)

# NearMiss-2
nm2 = NearMiss(version=2, n_neighbors=3)
X_nm2, y_nm2 = nm2.fit_resample(X, y)

# NearMiss-3
nm3 = NearMiss(version=3, n_neighbors=3, n_neighbors_ver3=3)
X_nm3, y_nm3 = nm3.fit_resample(X, y)
```

**Parámetros de NearMiss:**

| Parámetro | Descripción |
|-----------|-------------|
| `version` | `1`, `2` o `3`. Versión del algoritmo. |
| `n_neighbors` | Número de vecinos para calcular distancia promedio. |
| `n_neighbors_ver3` | Vecinos para la fase 1 de NearMiss-3. |
| `sampling_strategy` | Ratio deseado (`'auto'`, float o dict). |

**Pipeline con Scikit-learn:**

```python
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

# Pipeline: NearMiss + SVM
pipeline = make_pipeline(
    NearMiss(version=1, n_neighbors=3),
    SVC(kernel='rbf')
)

# Validación cruzada
scores = cross_validate(
    pipeline, X, y,
    scoring='average_precision',
    cv=5
)
```

---

## Capítulo 05 — ColumnTransformer en sklearn

### ¿Qué es ColumnTransformer?

`ColumnTransformer` es una clase de scikit-learn que permite aplicar **transformaciones diferentes a distintas columnas** del dataset simultáneamente.

**Problema que resuelve:** en datasets reales hay mezcla de tipos que requieren tratamientos distintos:

| Tipo de dato | Transformación requerida |
|---|---|
| Numéricas | Imputación + escalado |
| Categóricas nominales | One-Hot Encoding |
| Categóricas ordinales | Ordinal Encoding |

**Ventajas clave:**
- **Transformación selectiva** — preprocesamiento específico por tipo de dato.
- **Integración con Pipeline** — flujo de trabajo limpio y reproducible.
- **Evita data leakage** — fit en train, transform en test.

---

### Flujo de Trabajo: Tradicional vs ColumnTransformer

| ❌ Sin ColumnTransformer | ✅ Con ColumnTransformer |
|--------------------------|--------------------------|
| Separar columnas manualmente | Definir transformadores y columnas |
| Aplicar transformadores individuales | Un solo `fit_transform` |
| Concatenar resultados manualmente | Automáticamente se aplica a test |
| Repetir para train y test | Integrado en Pipeline |
| Riesgo de inconsistencias | Código limpio y mantenible |

**Sintaxis básica:**

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ],
    remainder='passthrough'
)
```

---

### Arquitectura de ColumnTransformer

**Formato de transformadores:** lista de tuplas `('nombre', transformer, columns)`

| Elemento | Descripción |
|----------|-------------|
| `'nombre'` | Identificador del transformador |
| `transformer` | Objeto transformer o Pipeline |
| `columns` | Lista de nombres o índices de columnas |

**Parámetro `remainder`:**

| Valor | Comportamiento |
|-------|----------------|
| `'drop'` (default) | Elimina columnas no transformadas |
| `'passthrough'` | Mantiene columnas sin modificar |
| `transformer` | Aplica transformador a columnas restantes |

**Tipos de selección de columnas:**
- Por nombre: `['age', 'income']`
- Por índice: `[0, 2, 5]`
- Slice: `slice(1, 5)`
- Boolean mask: `[True, False, True]`
- Selector: `make_column_selector(dtype_include='number')`

**Transformadores más comunes:**

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Numéricas
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categóricas
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

---

### Pipeline Completo con ColumnTransformer

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Dataset con tipos mixtos
X = pd.DataFrame({
    'age':    [25, 30, None, 35],
    'income': [50000, 60000, 55000, None],
    'city':   ['NYC', 'LA', 'NYC', 'Chicago'],
    'gender': ['M', 'F', 'F', 'M']
})

# Definir columnas
numeric_features     = ['age', 'income']
categorical_features = ['city', 'gender']

# Transformadores
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Pipeline completo
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Entrenar y predecir
clf.fit(X_train, y_train)
print(f"Score: {clf.score(X_test, y_test):.3f}")
```

---

### Integración Completa: Remuestreo + Preprocesamiento

> **Importante:** cuando se combina SMOTE/NearMiss con ColumnTransformer, se debe usar `imblearn.pipeline.Pipeline` en lugar de `sklearn.pipeline.Pipeline`.

| Razón | Detalle |
|-------|---------|
| `imblearn.Pipeline` | Aplica resampling después de los transformadores |
| `sklearn.Pipeline` | No soporta `fit_resample` |
| SMOTE | Debe aplicarse **solo** en datos de entrenamiento |

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Pipeline: Preprocesamiento → SMOTE → Modelo
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),          # ColumnTransformer
    ('smote', SMOTE(sampling_strategy=0.5)),
    ('classifier', RandomForestClassifier())
])

# Validación cruzada (SMOTE solo en training folds)
scores = cross_val_score(
    pipeline, X, y, cv=5,
    scoring='average_precision'
)
```

**Orden correcto de operaciones:**

1. **Preprocesamiento** — Imputación, escalado, encoding.
2. **Remuestreo** — SMOTE o NearMiss.
3. **Modelo** — Clasificador.

**Flujo End-to-End:**

1. Cargar datos y separar X, y.
2. Definir columnas numéricas y categóricas.
3. Crear `ColumnTransformer`.
4. Integrar en `imblearn.Pipeline` con SMOTE.
5. Validación cruzada con métricas apropiadas.
6. Evaluar en test set (sin SMOTE).

> **Regla de oro:** Nunca apliques SMOTE antes de dividir train/test. Siempre dentro de validación cruzada.

---

## Conclusiones y Mejores Prácticas

1. **Usa métricas apropiadas** — En datos desbalanceados, nunca uses accuracy. Prefiere Precision-Recall AUC cuando la clase positiva sea rara (< 10%). ROC-AUC es adecuado para datos más balanceados.

2. **Elige la técnica correcta** — Usa SMOTE cuando necesites más datos minoritarios. Usa NearMiss cuando el dataset sea grande y quieras reducirlo. Combina ambas para optimizar resultados.

3. **Preprocesamiento robusto** — Usa ColumnTransformer para manejar datos heterogéneos. Integra todo en pipelines para evitar fugas de datos y asegurar reproducibilidad.

4. **Validación correcta** — Aplica SMOTE/NearMiss solo en training, nunca en test. Usa `imblearn.Pipeline` dentro de validación cruzada para evitar overfitting.

5. **Hiperparámetros óptimos** — Para SMOTE: `k=5`, oversampling 100–400%. Para NearMiss: prueba las 3 versiones. Ratio final recomendado: 1:1 a 1:4 (mayoría:minoría).

---

## Recursos Adicionales

- Documentación imbalanced-learn: [imbalanced-learn.org](https://imbalanced-learn.org)
- Paper SMOTE original: Chawla et al. (2002)
- Scikit-learn User Guide: ColumnTransformer

---

> *"El manejo adecuado del desbalance es clave para el éxito de modelos de ML en aplicaciones reales."*
