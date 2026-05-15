# Transformación de Datos para Machine Learning

**Curso de Inteligencia Artificial**  
Centro Universitario de Guadalajara — Licenciatura en Inteligencia Artificial  
Febrero 2026

---

## Agenda

| # | Tema | Descripción |
|---|------|-------------|
| 01 | Introducción a la Transformación de Datos | Importancia crítica del preprocesamiento en ML y estadísticas sobre el tiempo dedicado por los científicos de datos |
| 02 | Tipos de Datos en Machine Learning | Clasificación de datos numéricos, categóricos y temporales con ejemplos prácticos de cada categoría |
| 03 | Manejo de Valores Faltantes | Mecanismos MCAR, MAR, MNAR y estrategias de imputación desde métodos simples hasta deep learning |
| 04 | Escalado y Normalización | Comparación de StandardScaler, MinMaxScaler y RobustScaler con fórmulas y casos de uso |
| 05 | Codificación de Variables Categóricas | One-Hot vs Label Encoding y técnicas avanzadas para alta cardinalidad |

---

## 1. ¿Por Qué Transformar los Datos?

### Requisitos de los Algoritmos

Los algoritmos de ML requieren datos numéricos y normalizados. Las variables categóricas deben codificarse y las características deben estar en escalas comparables para evitar sesgos.

### Escalas Comparables

Una característica con rango 0–100,000 dominará sobre otra con rango 0–1. El escalado asegura que todas las características contribuyan equitativamente al modelo.

### Calidad Determina Éxito

> *Garbage in, garbage out.* Un modelo con datos bien transformados puede superar a algoritmos más complejos con datos crudos.

### Estadísticas Clave

| Métrica | Valor |
|---------|-------|
| Tiempo de un Data Scientist dedicado a preprocesamiento | **80%** |
| Mejora en rendimiento con transformaciones adecuadas | **20–40%** |
| Convergencia más rápida en gradient descent | **3–5×** |

---

## 2. Tipos de Datos en Machine Learning

### Numéricos Continuos

Valores que pueden tomar cualquier número dentro de un rango, incluyendo decimales.

**Ejemplos:**
- Salario: $45,000.50
- Temperatura: 23.7 °C
- Peso: 72.5 kg

**Tratamiento:** Estandarización o normalización según distribución.

---

### Numéricos Discretos

Valores enteros con conteo finito o infinito numerable.

**Ejemplos:**
- Número de hijos: 0, 1, 2, 3
- Cantidad de productos: 5
- Años de experiencia: 7

**Tratamiento:** Puede tratarse como continuo o categórico ordinal.

---

### Categóricos Nominales

Categorías sin orden inherente. Cada valor es distinto pero no mayor o menor.

**Ejemplos:**
- Color: Rojo, Azul, Verde
- País: México, España, USA
- Género: Masculino, Femenino

**Tratamiento:** One-Hot Encoding o embeddings.

---

### Categóricos Ordinales

Categorías con orden natural o jerarquía inherente.

**Ejemplos:**
- Nivel educativo: Primaria < Secundaria < Universidad
- Satisfacción: Bajo < Medio < Alto
- Talla: S < M < L < XL

**Tratamiento:** Label Encoding preservando orden.

---

## 3. Manejo de Valores Faltantes

### Mecanismos de Datos Faltantes

| Mecanismo | Descripción |
|-----------|-------------|
| **MCAR** — Missing Completely at Random | La ausencia es completamente aleatoria, independiente de los datos observados y no observados. |
| **MAR** — Missing at Random | La ausencia depende de los datos observados pero no de los valores faltantes en sí. |
| **MNAR** — Missing Not at Random | La ausencia depende del valor faltante. Más difícil de manejar. |

> **Consejo Clave:** Entender el mecanismo es crucial. Los métodos simples funcionan bien para MCAR, pero MNAR requiere técnicas avanzadas o modelado explícito del mecanismo de ausencia.

### Estrategias de Imputación

| Tipo | Método | Descripción |
|------|--------|-------------|
| Simple | Estadísticos Básicos | Media, mediana, moda. Rápidos pero ignoran relaciones entre variables. |
| ML | KNN Imputation | Usa valores de los k vecinos más cercanos. Captura relaciones locales. |
| ML | Iterative Imputer (MICE) | Modela cada característica con valores faltantes como función de otras características. |

---

## 4. Escalado y Normalización

### StandardScaler

**Fórmula:** `z = (x - μ) / σ`

- Media = 0, Desviación estándar = 1
- Ideal para datos con distribución normal
- Sensible a outliers

**Usar con:** Regresión lineal, SVM, KNN, PCA, Redes neuronales.

---

### MinMaxScaler

**Fórmula:** `x' = (x - min) / (max - min)`

- Rango fijo [0, 1] o [-1, 1]
- Preserva la forma de la distribución
- Altamente sensible a outliers

**Usar con:** Redes neuronales, CNN, cuando se requiere rango acotado.

---

### RobustScaler

**Fórmula:** `x' = (x - mediana) / IQR`

- Usa mediana e IQR (Q3 - Q1)
- Robusto a outliers
- Ideal para distribuciones sesgadas

**Usar con:** Datos con outliers, distribuciones no normales.

---

### Guía de Selección

| Situación | Scaler recomendado |
|-----------|--------------------|
| Sin outliers + distribución normal | StandardScaler |
| Rango acotado requerido | MinMaxScaler |
| Con outliers / distribución sesgada | RobustScaler |

---

## 5. Codificación de Variables Categóricas

### One-Hot Encoding

Crea una columna binaria para cada categoría. Valor 1 indica presencia, 0 ausencia.

**Ejemplo:**

| País | MX | USA | ES |
|------|----|-----|----|
| México | 1 | 0 | 0 |

- ✅ No impone orden falso, interpretable
- ❌ Aumenta dimensionalidad rápidamente

---

### Label Encoding

Asigna un número entero único a cada categoría. Mantiene una sola columna.

**Ejemplo:** Bajo → 0 | Medio → 1 | Alto → 2

- ✅ Compacto, eficiente computacionalmente
- ❌ Puede implicar orden falso para nominales

---

### Técnicas Avanzadas

| Técnica | Descripción |
|---------|-------------|
| **Frequency/Count Encoding** | Reemplaza categoría con su frecuencia en el dataset. Una columna, compacto. |
| **Binary Encoding** | Convierte categorías a representación binaria. log₂(N) columnas vs N en one-hot. |
| **Target Encoding** | Reemplaza con media del target para esa categoría. Requiere validación cruzada para evitar leakage. |

### Guía de Selección

| Caso | Técnica |
|------|---------|
| Nominal + pocas categorías | One-Hot |
| Ordinal | Label |
| Alta cardinalidad | Target / Binary |

---

## 6. Pipeline de Preprocesamiento

### Arquitectura del Pipeline

1. **SimpleImputer** — Manejo de valores faltantes
2. **ColumnTransformer** — Aplicar transformaciones específicas por columna
3. **StandardScaler / OneHotEncoder** — Escalado y codificación
4. **Modelo ML** — Entrenamiento final

### Ventajas del Pipeline

- **Evita Data Leakage:** Transformaciones aprendidas solo en train
- **Reproducibilidad:** Mismas transformaciones en train/test/producción
- **Validación Cruzada:** Pipeline completo en cada fold
- **Mantenibilidad:** Código limpio y modular

### Ejemplo en Python

```python
# Importar librerías
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Definir columnas
numeric_features = ['edad', 'ingreso', 'score']
categorical_features = ['pais', 'genero']

# Pipeline numérico
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline categórico
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinar con ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline completo
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Entrenar
clf.fit(X_train, y_train)

# Predecir (aplica automáticamente transformaciones)
y_pred = clf.predict(X_test)
```

---

## 7. Mejores Prácticas

1. **Exploración Primero** — Siempre realizar EDA (Exploratory Data Analysis) antes de transformar. Entender distribuciones, outliers, correlaciones y tipos de datos.

2. **Validación Cruzada** — Nunca ajustar transformaciones en todo el dataset. Usar pipelines para aplicar transformaciones aprendidas solo en datos de entrenamiento.

3. **Experimentación** — No existe receta única. Probar diferentes técnicas y medir impacto en métricas de validación. Documentar decisiones.

4. **Automatización** — Usar pipelines de scikit-learn para encapsular todo el preprocesamiento. Facilita reproducibilidad y despliegue.

5. **Considerar el Modelo** — Las transformaciones dependen del algoritmo. Árboles son menos sensibles al escalado que SVM o redes neuronales.

---

> **Recuerda:** *"La calidad del preprocesamiento determina el límite superior del rendimiento del modelo. Un algoritmo sofisticado no puede compensar datos mal preparados."*

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

# Perceptrón & Redes Neuronales

**Inteligencia Artificial**  
Un viaje al corazón de la Inteligencia Artificial

---

## Agenda

| # | Tema | Descripción |
|---|------|-------------|
| 01 | La Neurona Artificial | La unidad fundamental que da vida a las redes neuronales, inspirada en el cerebro biológico. |
| 02 | Suma Ponderada y Pesos | El motor matemático: cómo las neuronas procesan información mediante pesos y bias. |
| 03 | Puertas Lógicas AND y OR | Demostrando el poder del aprendizaje lineal con ejemplos prácticos visuales. |
| 04 | El Problema del XOR | Cuando una sola neurona no es suficiente: el desafío que revolucionó el campo. |
| 05 | Funciones de Activación | Step, Sigmoid y ReLU: comparativa visual de las funciones que introducen no linealidad al modelo. |

---

## Capítulo 01 — La Neurona Artificial

### ¿Qué es una Neurona Artificial?

Una neurona artificial es la **unidad fundamental** de las redes neuronales artificiales. Es un modelo matemático inspirado en el funcionamiento de las neuronas biológicas del cerebro humano.

- **Procesamiento de información:** recibe, procesa y transmite información en sistemas de IA, emulando de manera simplificada el comportamiento de las neuronas biológicas.
- **Analogía biológica:** al igual que las neuronas biológicas se comunican mediante sinapsis, las neuronas artificiales se conectan mediante **pesos** que determinan la fuerza de la conexión.

---

### Anatomía de una Neurona Artificial (Perceptrón)

```
x₁ ──w₁──┐
x₂ ──w₂──┤──► Σ + b ──► f(x) ──► y
x₃ ──w₃──┘
```

**Ecuación:**

```
y = f(Σ(wᵢ · xᵢ) + b)
```

| Componente | Descripción |
|---|---|
| **Entradas (xᵢ)** | Múltiples valores numéricos provenientes de datos iniciales u otras neuronas. |
| **Pesos (wᵢ)** | Determinan la importancia de cada entrada. Son los parámetros de aprendizaje. |
| **Función de activación f(x)** | Introduce no linealidad, permitiendo aprender relaciones complejas. |
| **Salida (y)** | Resultado final; puede ser entrada de otra neurona o la predicción del modelo. |

---

## Capítulo 02 — Suma Ponderada y Pesos

### La Fórmula Central

```
z = w₁·x₁ + w₂·x₂ + w₃·x₃ + ... + wₙ·xₙ + b

z = Σ(wᵢ · xᵢ) + b
```

Cada entrada `xᵢ` se multiplica por su peso `wᵢ`, se suman todos los productos y se añade el bias `b`.

**Ejemplo práctico:**

| Parámetro | Valores |
|-----------|---------|
| Entradas | x₁ = 2, x₂ = 3, x₃ = 1 |
| Pesos | w₁ = 0.5, w₂ = −1, w₃ = 2 |
| Bias | b = 0.5 |

```
z = (0.5×2) + (−1×3) + (2×1) + 0.5
z = 1 − 3 + 2 + 0.5 = 0.5
```

---

### ¿Qué significan los pesos?

| Tipo de peso | Efecto |
|---|---|
| **Peso alto** | Fuerte influencia de la entrada en la salida. |
| **Peso bajo** | Influencia menor de la entrada. |
| **Peso negativo** | Influencia inhibidora: reduce la activación. |

> **Aprendizaje:** los pesos son los parámetros que la red ajusta iterativamente durante el entrenamiento para minimizar el error entre predicción y valor real.

---

### El Sesgo (Bias)

El bias `b` es un parámetro adicional que controla la facilidad con que una neurona se activa, independientemente de las entradas.

**Analogía con regresión lineal:**

```
y = mx + b
        ↑
   intersección con el eje Y
```

Sin bias, la recta de decisión siempre pasaría por el origen (0, 0).

| Escenario | Comportamiento |
|---|---|
| **Bias alto** | Requiere entrada más alta para activarse. |
| **Bias bajo** | La activación es más fácil. |
| **Con bias** | La recta de decisión puede desplazarse para ajustarse mejor a los datos. |
| **Sin bias** | La salida siempre pasa por el origen, limitando el poder del modelo. |

> **Ejemplo real:** la altura de una persona no es cero incluso si su ingesta calórica fuera cero. El bias permite capturar este tipo de relaciones.

---

## Capítulo 03 — Puertas Lógicas con Perceptrón

### Puerta Lógica AND

La salida es `1` **solo cuando ambas entradas son 1**. Es un problema linealmente separable.

| x₁ | x₂ | AND |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 0   |
| 1  | 0  | 0   |
| 1  | 1  | **1** |

> El perceptrón puede encontrar una línea recta que separa perfectamente los puntos con salida 0 de los puntos con salida 1.

---

### Puerta Lógica OR

La salida es `1` **cuando al menos una entrada es 1**. También linealmente separable.

| x₁ | x₂ | OR |
|----|----|----|
| 0  | 0  | 0  |
| 0  | 1  | **1** |
| 1  | 0  | **1** |
| 1  | 1  | **1** |

> Al igual que AND, OR es linealmente separable. El perceptrón aprende diferentes pesos para implementar esta función.

---

## Capítulo 04 — El Problema del XOR

### El Desafío del XOR

La salida es `1` **solo cuando exactamente una entrada es 1**.

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | **1** |
| 1  | 0  | **1** |
| 1  | 1  | 0   |

**El problema:** los puntos positivos `(0,1)` y `(1,0)` **no pueden separarse** de los negativos `(0,0)` y `(1,1)` con una sola línea recta.

---

### ¿Por Qué Falla el Perceptrón Simple?

El perceptrón simple solo puede generar **fronteras de decisión lineales:**

- Líneas rectas en 2D
- Planos en 3D
- Hiperplanos en n-D

**El "Invierno de la IA":** en 1969, Minsky y Papert demostraron esta limitación en su libro *Perceptrons*, contribuyendo a una fase de desconfianza hacia las redes neuronales.

---

### La Solución: Perceptrones Multicapa (MLP)

La solución llegó con los MLP en los años 80, mediante tres ideas clave:

| Paso | Concepto | Descripción |
|------|----------|-------------|
| 1 | **Múltiples capas** | Combinar neuronas en capas ocultas. |
| 2 | **No linealidad** | Funciones de activación no lineales. |
| 3 | **Backpropagation** | Algoritmo de entrenamiento eficiente. |

---

## Capítulo 05 — Funciones de Activación

### ¿Qué es una Función de Activación?

La función de activación introduce **no linealidad** en el modelo. Determina si una neurona debe activarse o permanecer inactiva.

| Sin función de activación | Con función de activación |
|---|---|
| Red neuronal ≡ Regresión lineal simple | La red puede aprender relaciones complejas |

**Proceso completo:**

```
1. Suma ponderada:   z = Σ(wᵢ · xᵢ) + b
2. Activación:       a = f(z)
3. Salida final:     a
```

**¿Por qué es importante?**

- Sin no linealidad, una red con múltiples capas sería equivalente a una sola capa.
- Con no linealidad, la red puede aprender relaciones arbitrariamente complejas.
- Permite aproximar cualquier función continua (**Teorema de Aproximación Universal**).

---

### Comparativa de Funciones de Activación

#### Step Function

```
f(x) = 0  si x < 0
f(x) = 1  si x ≥ 0
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Simple y fácil de entender | No diferenciable en x = 0 |
| Usada en el perceptrón original | Salida binaria limitada |

---

#### Sigmoid

```
σ(x) = 1 / (1 + e^(−x))
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Suave y diferenciable | Gradientes desvanecientes (*vanishing gradients*) |
| Salida interpretable como probabilidad (0, 1) | No centrada en cero |

---

#### ReLU (Rectified Linear Unit)

```
f(x) = max(0, x)
```

| ✅ Ventajas | ❌ Desventajas |
|---|---|
| Computacionalmente eficiente | Problema de neuronas "muertas" (x < 0) |
| Mitiga gradientes desvanecientes | — |
| Estándar en deep learning moderno | — |

---

### Resumen Comparativo

| Función | Fórmula | Rango de salida | Uso típico |
|---------|---------|----------------|------------|
| **Step** | `0 si x<0, 1 si x≥0` | {0, 1} | Perceptrón clásico |
| **Sigmoid** | `1 / (1 + e^−x)` | (0, 1) | Clasificación binaria (capa de salida) |
| **ReLU** | `max(0, x)` | [0, ∞) | Capas ocultas en deep learning |

---

## Conclusión

Las redes neuronales han revolucionado la inteligencia artificial. Desde el humilde perceptrón hasta las complejas arquitecturas de deep learning actuales, cada avance nos acerca más a máquinas verdaderamente inteligentes.

| Etapa | Acción |
|-------|--------|
| **Fundamentos** | Entender los conceptos básicos del perceptrón y las redes neuronales. |
| **Práctica** | Implementar lo aprendido con ejemplos reales. |
| **Innovación** | Construir el futuro sobre estos cimientos. |

> *"Las redes neuronales han revolucionado la inteligencia artificial. Desde el humilde perceptrón hasta las complejas arquitecturas de deep learning de hoy, cada avance nos acerca más a máquinas verdaderamente inteligentes."*

# Redes Neuronales — 8-Bit Edition

**Inteligencia Artificial**  
Centro Universitario de Guadalajara — Licenciatura en Inteligencia Artificial  
*Una aventura práctica, visual y divertida en el mundo de la IA*

---

## Menú Principal

| Nivel | Tema | Descripción |
|-------|------|-------------|
| 01 | ¿Qué es una Red Neuronal? | El cerebro artificial: conceptos fundamentales y analogías |
| 02 | De la Biología al Silicio | Comparando neuronas biológicas y artificiales |
| 03 | La Neurona Artificial | El Perceptrón: el modelo matemático básico |
| 04 | Arquitectura de una Red | Capas, conexiones y estructura multicapa |
| 05 | Funciones de Activación | Sigmoide, ReLU, Tanh: el pulso de la red |
| 06 | Proceso de Entrenamiento | Forward, Backpropagation y optimización |
| 07 | Tipos de Redes Neuronales | CNN, RNN, LSTM, GAN: especializaciones |
| 08 | Aplicaciones Reales | Casos de uso en el mundo actual |
| BOSS | Demo Práctica: XOR + Recursos | Código Python en acción y próximos pasos |

---

## World 1-1 — Nivel 1: Fundamentos

### Level 01 — ¿Qué es una Red Neuronal?

Una red neuronal es un sistema complejo de nodos interconectados (neuronas) que procesan y transmiten información, análogo a un cerebro artificial.

**Componentes clave:**

- **Neuronas (Nodos):** Unidades de procesamiento.
- **Conexiones:** Pesos y sesgos.
- **Capas:** Entrada, ocultas y salida.
- **Activación:** Funciones no lineales.

**Arquitectura básica:**

```
[CAPA DE ENTRADA]
  Recibe los datos iniciales (píxeles, números, texto)
          ▼
[CAPAS OCULTAS]
  Procesan y extraen patrones complejos
          ▼
[CAPA DE SALIDA]
  Produce el resultado final (predicción, clasificación)
```

---

### Level 02 — De la Biología al Silicio

| Componente biológico | Analogía artificial |
|---|---|
| **Dendritas** — Reciben señales de otras neuronas | Entradas de datos (x₁, x₂, ...) |
| **Cuerpo celular** — Procesa la información recibida | Función de activación |
| **Axón** — Transmite la señal a otras neuronas | Salida procesada |
| **Sinapsis** — Conexión con neurotransmisores | Pesos sinápticos (w₁, w₂, ...) |

| | Neurona Biológica | Neurona Artificial |
|---|---|---|
| Escala | ~100 mil millones de neuronas | Millones de operaciones/segundo |
| Consumo | ~12 W (menos que una bombilla) | 150 W+ (laptop típica) |

---

### Level 03 — La Neurona Artificial: El Perceptrón

**Diagrama del Perceptrón:**

```
x₁ ──w₁──┐
x₂ ──w₂──┼──► Σ(wᵢ·xᵢ) + b ──► f(z) ──► y
x₃ ──w₃──┘
```

**La fórmula mágica:**

```
z = Σ(wᵢ · xᵢ) + b    (suma ponderada + sesgo)
y = f(z)               (función de activación)
```

**Ejemplo práctico — ¿Debería salir a jugar?**

| Entrada | Descripción | Peso |
|---------|-------------|------|
| x₁ = 1 | ¿Hace sol? (1 = sí) | w₁ = 0.7 (importante) |
| x₂ = 1 | ¿Tengo tiempo? (1 = sí) | w₂ = 0.5 (moderado) |

```
z = (0.7 × 1) + (0.5 × 1) + (−0.2) = 1.0

Si f(z) > 0.5 → ¡SALIR A JUGAR!
```

---

## World 1-2 — Nivel 2: Arquitectura

### Level 04 — Arquitectura de una Red Neuronal (MLP)

En un **Multilayer Perceptron**, cada neurona se conecta con **todas** las neuronas de la capa siguiente (conexiones densas).

| Capa | Función | Ejemplo |
|------|---------|---------|
| **Capa de Entrada** | Recibe los datos iniciales; no realiza procesamiento. | Imagen 28×28 px = 784 neuronas |
| **Capas Ocultas** | Procesan y extraen patrones; cada capa aprende características más abstractas. | 1 o más capas (Deep Learning = muchas capas) |
| **Capa de Salida** | Produce el resultado final. | Clasificación binaria: 1 neurona; multiclase: N neuronas; regresión: 1 neurona continua |

---

### Level 05 — Funciones de Activación: El Pulso

#### Sigmoid (σ)

```
f(x) = 1 / (1 + e⁻ˣ)
```

| ✅ | ⚠️ |
|---|---|
| Rango: 0 a 1 (probabilidades) | Gradiente desvaneciente |
| Uso: Clasificación binaria | — |

#### Tanh

```
f(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)
```

| ✅ | ⚠️ |
|---|---|
| Rango: −1 a 1 (centrada en 0) | También sufre gradiente desvaneciente |
| Uso: Capas ocultas, RNN | — |

#### ReLU ⭐

```
f(x) = max(0, x)
```

| ✅ | ⚠️ |
|---|---|
| Rango: 0 a +∞ (no acotada) | Neuronas "muertas" para x < 0 |
| No sufre gradiente desvaneciente | — |
| Estándar en Deep Learning y CNN | — |

#### Softmax

```
f(xᵢ) = eˣⁱ / Σeˣʲ    → [0.1, 0.7, 0.2]
```

| ✅ | Uso |
|---|-----|
| Rango: Probabilidades (0–1) | Clasificación multiclase (capa de salida) |
| Suma de probabilidades = 1 | — |

---

## World 1-3 — Nivel 3: Especialización

### Level 06 — El Proceso de Entrenamiento

**Ciclo de entrenamiento:**

```
1. FORWARD PROPAGATION
   Los datos fluyen de la entrada hacia la salida.
   Cada neurona calcula su salida y la pasa a la siguiente capa.
              ▼
2. CÁLCULO DEL ERROR
   Se compara la predicción con el valor real
   mediante una función de pérdida (ej. MSE).
              ▼
3. BACKPROPAGATION
   El error se propaga hacia atrás, calculando cuánto
   contribuyó cada peso al error total (gradientes).
              ▼
4. ACTUALIZACIÓN DE PESOS
   Los pesos se ajustan con gradiente descendiente
   para minimizar el error en la siguiente iteración.
```

**Conceptos clave:**

| Concepto | Descripción | Valores típicos |
|----------|-------------|-----------------|
| **Época (Epoch)** | Un ciclo completo donde la red ve todos los datos de entrenamiento una vez. | — |
| **Lote (Batch)** | Subconjunto de datos procesados antes de actualizar pesos. | 32, 64, 128 |
| **Tasa de aprendizaje (η)** | Qué tan grandes son los ajustes de peso. | 0.001, 0.01, 0.1 |
| **Función de pérdida** | Mide qué tan lejos está la predicción del valor real. | MSE (regresión), Cross-Entropy (clasificación) |

**Fórmula del Gradiente Descendiente:**

```
w_new = w_old − η · (∂L/∂w)

η = tasa de aprendizaje
∂L/∂w = gradiente

Los pesos se ajustan en la dirección OPUESTA al gradiente
para minimizar el error.
```

---

### Level 07 — Tipos de Redes Neuronales

#### CNN — Redes Convolucionales

**Especialidad:** Procesamiento de imágenes y datos con estructura de cuadrícula.

**Componentes:** Capas convolucionales (filtros), Pooling (reducción dimensional), capas totalmente conectadas.

**Aplicaciones:** Reconocimiento facial, clasificación de imágenes, detección de objetos, vehículos autónomos.

**Arquitecturas famosas:** LeNet, AlexNet, VGG, ResNet.

---

#### RNN — Redes Recurrentes

**Especialidad:** Procesamiento de datos secuenciales (texto, series temporales).

**Características:** Memoria de estados anteriores, bucles de retroalimentación, procesamiento paso a paso.

**Aplicaciones:** NLP, traducción automática, generación de texto, predicción de series temporales.

> **Problema:** Gradientes desvanecientes en secuencias largas.

---

#### LSTM — Long Short-Term Memory

**Mejora sobre RNN:** Resuelve el problema de gradientes desvanecientes con celdas de memoria.

| Puerta (Gate) | Función |
|---|---|
| **Forget Gate** | Qué información olvidar |
| **Input Gate** | Qué información recordar |
| **Output Gate** | Qué output generar |

**Aplicaciones:** Chatbots, análisis de sentimiento, predicción bursátil, reconocimiento de voz.

> **Variante:** GRU (más eficiente, menos parámetros).

---

#### GAN — Redes Generativas Adversariales

**Objetivo:** Generar datos nuevos y realistas (imágenes, texto, audio).

**Dos redes en competencia:**
- **Generador:** Crea datos falsos.
- **Discriminador:** Detecta datos falsos.

**Aplicaciones:** Generación de imágenes (DALL-E, Midjourney), deepfakes, aumento de datos, transferencia de estilo artístico.

---

### Level 08 — Aplicaciones en el Mundo Real

| Dominio | Tecnología | Ejemplos destacados | Dato clave |
|---|---|---|---|
| **Reconocimiento facial** | CNN | Face ID (Apple), seguridad en aeropuertos | Precisión 99%+ en condiciones óptimas |
| **Asistentes virtuales** | RNN / LSTM / Transformers | Siri, Alexa, ChatGPT, Claude | 85% de interacciones automatizadas para 2025 |
| **Vehículos autónomos** | CNN + Deep Learning | Tesla Autopilot, Waymo Driver, ADAS | Reducción de costos operativos: 40% |
| **Diagnóstico médico** | CNN + Computer Vision | IBM Watson, detección de cáncer de mama | IA mejora detección de cáncer en 94% de casos |
| **Sistemas de recomendación** | Deep Learning + Collaborative Filtering | Netflix, Spotify, Amazon | 80% del contenido visto en Netflix es recomendado |
| **Procesamiento de lenguaje** | Transformers / LLM | ChatGPT, Google Translate, análisis de sentimiento | GPT-4 tiene 1.76 billones de parámetros |

---

## World 1-4 — Nivel 4: Práctica

### Boss Level — Demo: Resolviendo el Problema XOR

El perceptrón simple **no puede resolver XOR**. Se necesita una red multicapa con capa oculta.

**Tabla de verdad XOR:**

| x₁ | x₂ | XOR |
|----|----|-----|
| 0  | 0  | 0   |
| 0  | 1  | 1   |
| 1  | 0  | 1   |
| 1  | 1  | 0   |

**Resultados del entrenamiento:**

| Métrica | Valor |
|---------|-------|
| Épocas de entrenamiento | 10,000 |
| Error final | < 0.01 |
| Precisión | 100% |

**Código Python — Red Neuronal XOR desde cero:**

```python
import numpy as np

# Datos de entrenamiento XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Arquitectura: 2 → 4 → 1
input_size, hidden_size, output_size = 2, 4, 1

# Inicializar pesos aleatorios
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Función sigmoide
sigmoid = lambda x: 1 / (1 + np.exp(-x))

# Entrenamiento: 10,000 épocas
for epoch in range(10000):
    # Forward propagation
    hidden = sigmoid(X.dot(W1))
    output = sigmoid(hidden.dot(W2))

    # Calcular error
    error = y - output

    # Backpropagation
    d_output = error * output * (1 - output)
    d_hidden = d_output.dot(W2.T) * hidden * (1 - hidden)

    # Actualizar pesos (learning rate = 0.5)
    W2 += hidden.T.dot(d_output) * 0.5
    W1 += X.T.dot(d_hidden) * 0.5

# Predicciones finales
print("XOR predicciones:", np.round(output))
```

---

### Bonus Level — Recursos y Próximos Pasos

#### Frameworks Populares

| Framework | Creador | Ideal para |
|-----------|---------|------------|
| **TensorFlow** | Google | Producción y deployment |
| **PyTorch** | Meta | Investigación y prototipado |
| **Keras** | (API sobre TF) | Alto nivel, fácil de usar |

#### Libros Recomendados

- *Deep Learning* — Goodfellow, Bengio, Courville
- *Neural Networks and Deep Learning* — Michael Nielsen
- *Hands-On Machine Learning* — Aurélien Géron

#### Cursos Online

- **Coursera — Deep Learning Specialization** (Andrew Ng / Stanford) — 5 cursos, muy completo.
- **Fast.ai — Practical Deep Learning** — Enfoque práctico, código primero.
- **YouTube — 3Blue1Brown** — Visualizaciones matemáticas increíbles.

#### Proyectos Prácticos Sugeridos

1. **Clasificador de imágenes** — Distinguir gatos vs. perros con CNN.
2. **Análisis de sentimiento** — Clasificar tweets positivos/negativos.
3. **Predicción de precios** — Forecasting de acciones o criptomonedas.
4. **Chatbot simple** — Bot de preguntas y respuestas con RNN.
5. **Generador de texto** — Crear poemas o historias con LSTM.

> **Consejo final:** ¡Practica, practica, practica! La mejor forma de aprender es construyendo proyectos propios.

---

## Game Completed!

| Estadística | Valor |
|-------------|-------|
| Niveles completados | 17 |
| Conocimiento adquirido | 100% |
| Posibilidades | ∞ |

*Ahora eres un HÉROE de las Redes Neuronales.*
