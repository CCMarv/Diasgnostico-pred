# Notas de Clase — Introducción a la Inteligencia Artificial
**Centro Universitario de Guadalajara | Licenciatura en Inteligencia Artificial**

---

## Índice

- [Unidad 1 — El Pipeline de ML y Procesamiento de Datos Avanzado](#unidad-1)
  - [1.1 Ingeniería del Flujo de Trabajo](#11-ingeniería-del-flujo-de-trabajo)
  - [1.2 Transformación Avanzada — Notebook: escalar_o_no_escalar](#12-transformación-avanzada)
  - [1.3 Gestión del Desbalance — Notebook: gestion_desbalance](#13-gestión-del-desbalance)
- [Unidad 2 — Aprendizaje Supervisado: Modelos No Lineales y Ensambles](#unidad-2)
  - [2.1 Support Vector Machines — Notebook: smv_fraude](#21-support-vector-machines-svm)
  - [2.2 Árboles de Decisión](#22-árboles-de-decisión)
  - [2.3 Métodos de Ensamble](#23-métodos-de-ensamble)
- [Unidad 3 — Aprendizaje No Supervisado](#unidad-3)
  - [3.1 Agrupamiento (Clustering)](#31-agrupamiento-clustering)
  - [3.2 Reducción de Dimensionalidad](#32-reducción-de-dimensionalidad)
- [Unidad 4 — Optimización e Introducción a Redes Neuronales](#unidad-4)
  - [4.1 Optimización de Hiperparámetros](#41-optimización-de-hiperparámetros)
  - [4.2 Introducción a Redes Neuronales](#42-introducción-a-redes-neuronales)
- [Referencias](#referencias)

---

# Unidad 1
# El Pipeline de ML y Procesamiento de Datos Avanzado

---

## 1.1 Ingeniería del Flujo de Trabajo

### El ciclo de vida del ML

Un proyecto de Machine Learning no comienza con el modelo: comienza con los datos. El ciclo de vida completo incluye recopilación y exploración de datos, preprocesamiento y transformación, selección y entrenamiento del modelo, evaluación y validación, y finalmente despliegue en producción.

La herramienta central para estructurar este flujo en scikit-learn es `sklearn.pipeline.Pipeline`, que encadena pasos de transformación y modelado en un objeto reutilizable. Su ventaja principal es garantizar que los datos de test nunca influyan en el proceso de entrenamiento.

### Data Leakage: el error silencioso más costoso

> **Definición:** el Data Leakage (fuga de datos) ocurre cuando información del conjunto de test contamina el proceso de entrenamiento, produciendo estimaciones de rendimiento artificialmente optimistas que no se replican en producción.

Kaufman, Rosset y Perlich (2012) formalizaron el problema en su trabajo fundacional *Leakage in Data Mining: Formulation, Detection, and Avoidance*, identificando que el leakage es una de las causas más frecuentes de modelos que funcionan perfectamente en evaluación pero fallan en el mundo real. Investigaciones posteriores han confirmado que el leakage durante el preprocesamiento —aplicar transformaciones globales antes del split train/test— es especialmente común y difícil de detectar (Kapoor & Narayanan, 2023).

**Formas comunes de leakage en preprocesamiento:**

| Acción problemática | Por qué produce leakage |
|---|---|
| Calcular la media/std con todo el dataset y luego escalar | El scaler "ve" la distribución del test set |
| Imputar valores faltantes con la media global | La media del test contamina la imputación del train |
| Aplicar SMOTE antes del split | Muestras sintéticas se generan con información del test |
| Codificar categorías con frecuencias del dataset completo | Las frecuencias del test filtran información al train |

**La solución: ajustar (`fit`) solo en train, transformar (`transform`) en todo**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),   # fit_transform en train, transform en test
    ('model',  LogisticRegression())
])

pipeline.fit(X_train, y_train)     # el scaler solo aprende de X_train
pipeline.score(X_test, y_test)     # aplica transform (no fit) a X_test
```

---

## 1.2 Transformación Avanzada

> **📓 Notebook asociado:** `escalar_o_no_escalar.ipynb` | Dataset: `buro_credito.csv` (5,383 registros)

### Contexto teórico

#### Imputación multivariable (KNNImputer)

La imputación simple con la media o la mediana asume que los valores faltantes son independientes del resto de las variables, lo que raramente es cierto en datasets reales. El **KNNImputer** reemplaza cada valor faltante con el promedio ponderado de los *k* vecinos más similares en el espacio de las demás variables:

```
x̂_ij = (1/k) · Σ x_lj  para los k vecinos más cercanos l a i
```

Esto preserva las correlaciones entre variables y produce imputaciones más plausibles cuando los datos tienen estructura (Little & Rubin, 2002, *Statistical Analysis with Missing Data*).

#### Transformaciones de potencia: Yeo-Johnson

Muchos algoritmos asumen que las variables siguen una distribución aproximadamente normal. La transformación de Yeo-Johnson (Yeo & Johnson, 2000) es una generalización de la transformación Box-Cox que acepta valores negativos:

```
        (x + 1)^λ − 1) / λ          si λ ≠ 0, x ≥ 0
φ(x,λ) = ln(x + 1)                  si λ = 0, x ≥ 0
        −(1 − x)^(2−λ) − 1)/(2−λ)  si λ ≠ 2, x < 0
        −ln(1 − x)                  si λ = 2, x < 0
```

Es especialmente útil antes de aplicar algoritmos sensibles a la distribución como regresión lineal, PCA o KNN.

#### Codificación robusta: TargetEncoder y HashingEncoder

El **TargetEncoder** reemplaza cada categoría con la media del target para esa categoría. Es eficaz para variables de alta cardinalidad, pero requiere validación cruzada interna para evitar leakage (la media del target en train nunca debe calcularse con el test). El **HashingEncoder** proyecta categorías a un espacio de dimensión fija mediante una función hash, siendo útil cuando hay miles de categorías únicas.

### ¿Cuándo escalar los datos?

#### Problema del notebook

Se predice si un cliente pagará o no un crédito bancario (*default / no default*). El objetivo central no es la predicción en sí, sino **comparar empíricamente el efecto del escalamiento** en modelos con diferentes sensibilidades a la escala de los datos.

Las tres variables de entrada tienen rangos muy distintos:

| Variable | Rango sin escalar |
|---|---|
| Puntaje crediticio | 0 – 7.5 |
| Ingresos mensuales (USD) | 1.4 – 6,997.8 |
| Monto del préstamo (USD) | 20,023 – 49,991 |

Sin escalamiento, la variable "Monto del préstamo" domina cualquier cálculo basado en distancias o gradientes, ignorando prácticamente las demás.

#### Tipos de escalamiento

| Técnica | Fórmula | Resultado | Sensible a outliers |
|---|---|---|---|
| **MinMaxScaler** | `x' = (x − min) / (max − min)` | Rango \[0, 1\] | Sí |
| **StandardScaler** | `z = (x − μ) / σ` | Media 0, desviación 1 | Moderado |
| **RobustScaler** | `x' = (x − mediana) / IQR` | Robusto | No |

En el notebook se usa **MinMaxScaler** (normalización), que escala todas las variables al rango \[0, 1\].

#### Regla crítica: fit solo en train

```python
scaler = MinMaxScaler()

xtr_s = scaler.fit_transform(xtr)  # aprende min/max del train
xvl_s = scaler.transform(xvl)      # aplica esos min/max aprendidos
xts_s = scaler.transform(xts)      # nunca vuelve a hacer fit
```

> Ajustar el scaler con datos de validación o test constituye Data Leakage: el modelo aprende implícitamente la distribución del test set antes de evaluarse sobre él.

#### Funciones implementadas

**`crear_in_out(datos)`** — Separa el DataFrame en entradas `X` (columnas 0–2) y salida `Y` (columna 3).

**`crear_train_val_test(X, Y)`** — Divide en tres sets usando `train_test_split` dos veces: 70% entrenamiento, 15% validación, 15% prueba.

**`escalar_datos(xtr, xvl, xts)`** — Aplica MinMaxScaler con la regla fit-only-on-train.

#### Experimento: efecto del escalamiento por tipo de modelo

**Árboles de clasificación (`DecisionTreeClassifier`)**

Los árboles construyen fronteras de decisión basadas en umbrales sobre variables individuales. Al escalar, el umbral se desplaza en la misma proporción → el resultado no cambia. El desempeño es **idéntico** con y sin escalamiento.

**Clasificador kNN (`KNeighborsClassifier`, k=3)**

kNN clasifica un punto por mayoría de votos entre sus *k* vecinos más cercanos, medidos por distancia euclidiana. Si las variables tienen escalas muy diferentes, las de mayor rango dominan el cálculo:

```
d(a, b) = √[(a₁−b₁)² + (a₂−b₂)² + (a₃−b₃)²]
```

Una variable con rango 50,000 contribuye ~7,000 veces más que una con rango 7.5. El escalamiento elimina este sesgo y generalmente **mejora** el desempeño del kNN.

**Redes Neuronales (MLP)**

El entrenamiento de redes neuronales usa gradiente descendiente, cuya convergencia depende de la escala de los gradientes. Con datos sin escalar, los gradientes pueden ser muy dispares entre capas, causando convergencia lenta o inestable. El escalamiento previo produce convergencia más rápida y estable.

#### Tabla resumen: sensibilidad al escalamiento

Un estudio sistemático de 12 técnicas de escalamiento sobre 14 algoritmos y 16 datasets (publicado en IEEE, 2025) confirmó empíricamente lo siguiente:

| Modelo | ¿Requiere escalamiento? | Razón |
|---|---|---|
| Árbol de decisión | No | Umbrales internos se adaptan a la escala |
| Random Forest | No* | *Puede acelerar el entrenamiento |
| Gradient Boosting / XGBoost | No | Ídem árboles |
| **kNN** | **Sí** | Cálculo de distancias euclidianas |
| **SVM** | **Sí** | Maximización del margen depende de la escala |
| **Redes Neuronales (MLP)** | **Sí** | Convergencia del gradiente descendiente |
| **Regresión logística / lineal** | **Sí** | Optimización convexa basada en gradientes |

> **Recomendación práctica:** escalar los datos independientemente del modelo elegido. Nunca perjudica y puede mejorar rendimiento y velocidad de entrenamiento.

---

## 1.3 Gestión del Desbalance

> **📓 Notebook asociado:** `gestion_desbalance.ipynb` | Dataset: `fraude.csv`

### Contexto teórico

#### El problema del desbalance de clases

Un dataset está desbalanceado cuando las clases no están aproximadamente igualadas en número. En dominios como detección de fraude, diagnóstico médico o detección de intrusiones, la clase de interés puede representar menos del 1% del total.

El problema fundamental es que los algoritmos de clasificación optimizan métricas globales como el accuracy. En un dataset 99:1, un clasificador que siempre predice la clase mayoritaria obtiene 99% de accuracy pero 0% de recall sobre la clase minoritaria — la que importa.

Chawla et al. (2002), en el paper fundacional *SMOTE: Synthetic Minority Over-sampling Technique* (Journal of Artificial Intelligence Research, 16, 321–357), formalizaron este problema y propusieron la primera solución basada en generación sintética de muestras que se volvería estándar en la industria.

#### Métricas correctas para datos desbalanceados

**La matriz de confusión:**

|  | Predicho Negativo | Predicho Positivo |
|--|---|---|
| **Real Negativo** | TN | FP |
| **Real Positivo** | FN | TP |

**Métricas relevantes:**

| Métrica | Fórmula | Interpretación |
|---|---|---|
| **Precision** | `TP / (TP + FP)` | De las predicciones positivas, ¿cuántas son correctas? |
| **Recall** | `TP / (TP + FN)` | De los positivos reales, ¿cuántos detectamos? |
| **F1-Score** | `2 · (P · R) / (P + R)` | Media armónica de Precision y Recall |
| **ROC-AUC** | Área bajo la curva ROC | Capacidad discriminativa general |
| **PR-AUC** | Área bajo la curva PR | Rendimiento enfocado en la clase positiva |

> **Regla práctica:** si la clase positiva representa menos del 10% del dataset, usar **Precision-Recall AUC**. Entre 20–50%, ROC-AUC es adecuado.

#### SMOTE: generación sintética de muestras

SMOTE (Chawla et al., 2002) genera muestras sintéticas de la clase minoritaria mediante interpolación lineal entre puntos reales, en lugar de simplemente duplicarlos:

```
x_new = xᵢ + λ · (xₙₙ − xᵢ),   λ ~ U(0, 1)
```

donde `xᵢ` es una muestra minoritaria y `xₙₙ` es uno de sus *k* vecinos más cercanos en la clase minoritaria. Esto expande las regiones de decisión y reduce el riesgo de overfitting que causaría la simple replicación.

#### NearMiss: undersampling inteligente

NearMiss selecciona qué muestras de la clase mayoritaria conservar, basándose en su distancia a la clase minoritaria. Existen tres versiones:

| Versión | Criterio de selección |
|---|---|
| **NearMiss-1** | Conserva mayoritarios con menor distancia promedio a los *k* minoritarios más cercanos |
| **NearMiss-2** | Conserva mayoritarios con menor distancia promedio a los *k* minoritarios más lejanos |
| **NearMiss-3** | Dos fases: primero selecciona *m* mayoritarios cercanos por cada minoritario; luego conserva los de mayor distancia promedio a la minoría |

### Implementación en el notebook

#### Librerías necesarias

```python
pip install imbalanced-learn seaborn
```

```python
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import NearMiss
from imblearn.pipeline        import Pipeline as ImbPipeline  # no sklearn
from sklearn.compose          import ColumnTransformer
from sklearn.ensemble         import RandomForestClassifier
from sklearn.metrics          import (classification_report,
                                      roc_auc_score,
                                      precision_recall_curve, auc)
```

#### Flujo paso a paso

**Paso 1 — Separar target**

```python
X = df.drop('es_fraude', axis=1)
y = df['es_fraude']
```

**Paso 2 — Split estratificado**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

`stratify=y` garantiza que la proporción de fraudes sea idéntica en train y test, evitando que el split aleatorio concentre todos los fraudes en un solo subset.

**Paso 3 — ColumnTransformer**

```python
num_features = ['monto', 'antiguedad_cuenta']
cat_features = ['tipo_tarjeta']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(), cat_features)
])
```

`ColumnTransformer` aplica transformaciones diferentes según el tipo de dato de cada columna, en una sola operación integrable en el pipeline.

**Paso 4 — Pipeline con SMOTE (oversampling)**

```python
pipeline_smote = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline_smote.fit(X_train, y_train)
```

**Paso 5 — Evaluación**

```python
y_pred  = pipeline_smote.predict(X_test)
y_probs = pipeline_smote.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC:              {roc_auc_score(y_test, y_probs):.4f}")

precision, recall, _ = precision_recall_curve(y_test, y_probs)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC: {pr_auc:.4f}")
```

**Paso 6 — Comparación con NearMiss (undersampling)**

```python
pipeline_nearmiss = ImbPipeline([
    ('preprocessor', preprocessor),
    ('nearmiss', NearMiss(version=1)),
    ('classifier', RandomForestClassifier(random_state=42))
])
pipeline_nearmiss.fit(X_train, y_train)
```

#### Conceptos clave

**¿Por qué `imblearn.Pipeline` y no `sklearn.Pipeline`?**

| | `sklearn.Pipeline` | `imblearn.Pipeline` |
|---|---|---|
| Soporta `fit_resample` | No | Sí |
| Aplica SMOTE solo en train | No | Sí |
| Correcto en validación cruzada | No | Sí |

Usar `sklearn.Pipeline` con SMOTE provoca que el remuestreo se aplique también sobre los datos de test → leakage. Investigaciones sobre pipelines y leakage (Kapoor & Narayanan, 2023; Ichwani et al., 2026) han documentado este error en numerosos estudios publicados con resultados inflados.

**SMOTE vs NearMiss**

| | SMOTE (Oversampling) | NearMiss (Undersampling) |
|---|---|---|
| Acción | Genera datos sintéticos minoritarios | Elimina datos mayoritarios seleccionados |
| Tamaño final | Dataset más grande | Dataset más pequeño |
| Riesgo | Overfitting si se aplica fuera del pipeline | Pérdida de información relevante |
| Cuándo usar | Dataset relativamente pequeño | Dataset muy grande donde el costo de memoria es relevante |

**Parámetros clave de SMOTE:**

| Parámetro | Descripción |
|---|---|
| `sampling_strategy` | Ratio final minoría/mayoría (ej. `0.3` → 30%) |
| `k_neighbors` | Vecinos usados para interpolar (default: 5) |
| `random_state` | Semilla para reproducibilidad |

> **Regla de oro:** nunca apliques SMOTE antes de dividir en train/test. Siempre dentro del pipeline, para que el remuestreo ocurra solo sobre los datos de entrenamiento en cada fold de validación cruzada.

---

# Unidad 2
# Aprendizaje Supervisado: Modelos No Lineales y Ensambles

---

## 2.1 Support Vector Machines (SVM)

> **📓 Notebook asociado:** `smv_fraude.ipynb` | Dataset: `fraude.csv`

### Contexto teórico

#### Origen y fundamento

Las Support Vector Machines fueron introducidas por Cortes y Vapnik en su paper seminal *Support-Vector Networks* (Machine Learning, 20, 273–297, 1995), con raíces en la teoría de aprendizaje estadístico de Vapnik-Chervonenkis. El algoritmo se fundamenta en el principio de **minimización del riesgo estructural**: en lugar de minimizar solo el error en el conjunto de entrenamiento (riesgo empírico), busca minimizar una cota superior del error de generalización.

La idea central es que, entre todos los hiperplanos que separan correctamente las dos clases, el que maximiza el margen entre ellas tendrá mayor capacidad de generalización.

#### El hiperplano de máximo margen

Dado un conjunto de puntos `{(xᵢ, yᵢ)}` con `yᵢ ∈ {−1, +1}`, un SVM busca el hiperplano `w·x + b = 0` que satisface:

```
Maximizar:  2 / ‖w‖          (el margen)
Sujeto a:   yᵢ(w·xᵢ + b) ≥ 1   para todo i
```

Equivalentemente, esto se formula como la minimización de `‖w‖²/2`, un problema de optimización convexa con solución única y global.

```
Clase positiva (+1)   ●  ●  ●
                              ← margen = 2/‖w‖ →
                       ───────────────────────────  ← hiperplano w·x + b = 0
                              ← margen = 2/‖w‖ →
Clase negativa (−1)               ● ● ●
```

**Los vectores de soporte** son los puntos que quedan exactamente sobre los márgenes (`yᵢ(w·xᵢ + b) = 1`). Son los únicos que determinan la posición del hiperplano; el resto de los puntos no influye en la solución.

#### SVM de margen suave (soft margin)

En datos no perfectamente separables, el SVM permite violaciones del margen controladas por variables de holgura `ξᵢ ≥ 0` y el parámetro de regularización **C**:

```
Minimizar:   ‖w‖²/2 + C · Σξᵢ
Sujeto a:    yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ
```

| Valor de C | Efecto sobre el margen | Riesgo |
|---|---|---|
| C alto (ej. 100) | Margen estrecho, penaliza fuertemente los errores | Overfitting |
| C = 1 (default) | Balance entre margen y errores de clasificación | Equilibrado |
| C bajo (ej. 0.01) | Margen amplio, tolera más errores | Underfitting |

#### El truco del kernel

Para datos no linealmente separables, el kernel trick proyecta los datos a un espacio de mayor dimensión donde sí son linealmente separables, sin calcular explícitamente esa proyección:

```
K(xᵢ, xⱼ) = φ(xᵢ) · φ(xⱼ)
```

| Kernel | Función | Uso típico |
|---|---|---|
| Lineal | `K(xᵢ, xⱼ) = xᵢ · xⱼ` | Datos linealmente separables |
| RBF (Gaussiano) | `K(xᵢ, xⱼ) = exp(−γ‖xᵢ − xⱼ‖²)` | Fronteras no lineales (más usado) |
| Polinomial | `K(xᵢ, xⱼ) = (xᵢ · xⱼ + r)^d` | Relaciones polinomiales |

### Implementación en el notebook

#### Problema a resolver

Se entrena un SVM con kernel lineal para detectar transacciones fraudulentas y se genera una **visualización 3D del hiperplano de separación** para desarrollar la intuición geométrica del modelo.

#### Flujo del notebook

**1 — Preparación de datos**

```python
df = pd.read_csv('fraude.csv')
X  = df[['monto', 'antiguedad_cuenta']].values
y  = df['es_fraude'].values
```

El modelo usa solo **dos características** para poder visualizar el hiperplano en 3D (features en ejes X-Y, función de decisión en eje Z).

**2 — Escalado (obligatorio en SVM)**

```python
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Los SVM son altamente sensibles a la escala porque la maximización del margen depende de distancias euclidianas. Sin escalar, la variable con mayor rango domina el cálculo del margen.

**3 — Entrenamiento del SVM lineal**

```python
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_scaled, y)
```

**4 — Construcción de la visualización 3D**

La ecuación del hiperplano `w₀·x + w₁·y + b = 0` se despeja en `z`:

```python
w  = svm_model.coef_[0]
b  = svm_model.intercept_[0]
zz = (-w[0] * xx - w[1] * yy - b)

# Hiperplano
ax.plot_surface(xx, yy, zz, alpha=0.3, color='gray')

# Puntos legítimos
ax.scatter(X_scaled[y==0, 0], X_scaled[y==0, 1], 0,
           color='blue', label='Legítimo (0)')

# Puntos fraude
ax.scatter(X_scaled[y==1, 0], X_scaled[y==1, 1], 0,
           color='red', label='Fraude (1)')

# Vectores de soporte
ax.scatter(svm_model.support_vectors_[:, 0],
           svm_model.support_vectors_[:, 1], 0,
           facecolors='none', edgecolors='black', label='Vectores de soporte')

ax.view_init(elev=20, azim=45)  # ajustar ángulo de cámara
```

**Elementos del gráfico:**

| Elemento visual | Representación |
|---|---|
| Plano gris translúcido | Hiperplano de decisión `w·x + b = 0` |
| Puntos azules | Transacciones legítimas (clase 0) |
| Puntos rojos | Fraudes (clase 1) |
| Círculos negros vacíos | Vectores de soporte |

#### Atributos útiles del modelo entrenado

| Atributo | Contenido |
|---|---|
| `svm_model.coef_` | Vector de pesos `w` (solo en kernel lineal) |
| `svm_model.intercept_` | Sesgo `b` del hiperplano |
| `svm_model.support_vectors_` | Coordenadas de los vectores de soporte |
| `svm_model.n_support_` | Número de vectores de soporte por clase |

---

## 2.2 Árboles de Decisión

### Contexto teórico

Los árboles de decisión son modelos que particionan recursivamente el espacio de características mediante reglas binarias de la forma `xⱼ ≤ t`. El algoritmo **CART** (Classification and Regression Trees, Breiman et al., 1984) es el más extendido y es el que utiliza scikit-learn.

#### Algoritmo de construcción (CART)

En cada nodo, el algoritmo evalúa todas las posibles divisiones `(xⱼ, t)` y selecciona la que minimiza la impureza resultante. Para clasificación, las métricas de impureza son:

**Índice de Gini:**

```
Gini(t) = 1 − Σₖ p²ₖ
```

donde `pₖ` es la proporción de la clase `k` en el nodo `t`. Un nodo puro tiene Gini = 0; un nodo perfectamente impuro (clases igualmente distribuidas) tiene Gini = 0.5.

**Entropía de Shannon:**

```
H(t) = −Σₖ pₖ log₂(pₖ)
```

Proviene de la teoría de la información (Shannon, 1948). Una distribución uniforme maximiza la entropía; una distribución pura la lleva a cero.

La ganancia de información de una división es:

```
IG = H(padre) − [Nᵢzq/N · H(izq) + Nᵈᵉʳ/N · H(der)]
```

#### Propiedades importantes

- **Invarianza al escalamiento:** los árboles son invariantes a transformaciones monótonas de las variables (incluyendo escalamiento), porque los umbrales de división se adaptan automáticamente. Esto los hace únicos entre los algoritmos de ML.
- **No linealidad:** pueden aprender fronteras de decisión arbitrariamente complejas (escaleras de hiperplanos).
- **Interpretabilidad:** cada path desde la raíz a una hoja es una regla de decisión legible.
- **Tendencia al overfitting:** sin regularización (profundidad máxima, mínimo de muestras por hoja), los árboles memorizan el conjunto de entrenamiento.

---

## 2.3 Métodos de Ensamble

### Contexto teórico

Los métodos de ensamble combinan múltiples modelos base para obtener predicciones más robustas. La justificación teórica viene del teorema de Condorcet (1785) y del principio de la "sabiduría de las multitudes": si los errores de los modelos individuales son independientes, su combinación reduce el error.

Breiman (1996, 2001) formalizó las dos familias principales con sus papers *Bagging Predictors* y *Random Forests*.

#### Bagging: Random Forests

**Bootstrap Aggregating (Bagging)** entrena cada árbol en una muestra aleatoria con reemplazo del dataset original (bootstrap sample). La predicción final es la moda (clasificación) o la media (regresión).

**Random Forests** añade una segunda fuente de aleatoriedad: en cada nodo, solo se considera un subconjunto aleatorio de `m` variables (típicamente `m = √p` para clasificación). Esto desorrela los árboles, reduciendo la varianza del ensamble.

```
Bias del ensamble    ≈ Bias de un árbol individual
Varianza del ensamble ≈ ρσ² + (1-ρ)σ²/B
```

donde `ρ` es la correlación entre árboles y `B` es el número de árboles. A mayor número de árboles y menor correlación, menor varianza del ensamble.

#### Boosting: Gradient Boosting

A diferencia del bagging (paralelo), el boosting es secuencial: cada nuevo modelo se entrena para corregir los errores del modelo anterior. Friedman (2001) formuló el **Gradient Boosting** como un algoritmo de descenso de gradiente en el espacio funcional:

```
F_m(x) = F_{m-1}(x) + η · h_m(x)
```

donde `h_m` es un árbol entrenado sobre los residuos (gradiente negativo de la función de pérdida) y `η` es la tasa de aprendizaje.

**XGBoost** (Chen & Guestrin, 2016) y **LightGBM** (Ke et al., 2017) son implementaciones optimizadas que incorporan regularización L1/L2, manejo eficiente de datos dispersos y técnicas de paralelización que los hacen entre 10–100× más rápidos que implementaciones ingenuas.

#### Complejidad computacional

| Modelo | Entrenamiento | Inferencia | Espacio |
|---|---|---|---|
| Árbol de decisión | O(n · p · log n) | O(profundidad) | O(nodos) |
| Random Forest (B árboles) | O(B · n · m · log n) | O(B · profundidad) | O(B · nodos) |
| Gradient Boosting (M iteraciones) | O(M · n · p · log n) | O(M · profundidad) | O(M · nodos) |

---

# Unidad 3
# Aprendizaje No Supervisado: Clustering y Dimensionalidad

---

## 3.1 Agrupamiento (Clustering)

### Contexto teórico

El clustering agrupa datos sin etiquetas en grupos (clusters) de tal forma que los elementos dentro de un grupo sean más similares entre sí que con los de otros grupos. No existe una definición única de "cluster": cada algoritmo implementa una noción diferente de similitud.

#### K-Means: Expectation-Maximization

K-Means (MacQueen, 1967; Lloyd, 1982) minimiza la suma de cuadrados intra-cluster:

```
J = Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²
```

El algoritmo alterna entre dos pasos:

- **E-step (Expectation):** asignar cada punto al centroide más cercano.
- **M-step (Maximization):** recalcular cada centroide como la media de los puntos asignados.

**Inicialización K-Means++** (Arthur & Vassilvitskii, 2007) selecciona los centroides iniciales con probabilidad proporcional a su distancia al centroide anterior, lo que reduce drásticamente la probabilidad de converger a mínimos locales malos. Está implementado por defecto en scikit-learn (`init='k-means++'`).

#### DBSCAN: clustering basado en densidad

DBSCAN (Ester et al., 1996) define clusters como regiones de alta densidad separadas por regiones de baja densidad. A diferencia de K-Means, puede detectar clusters de forma arbitraria y etiquetar puntos como ruido.

Dos parámetros controlan el comportamiento:

- **ε (epsilon):** radio de la vecindad de un punto.
- **MinPts:** mínimo de puntos en la ε-vecindad para que un punto sea considerado "núcleo".

Un punto es clasificado como núcleo, frontera o ruido según cuántos vecinos tiene dentro de ε.

#### Validación interna: ¿cuántos clusters elegir?

**Método del Codo:** graficar la inercia (J) en función de k y buscar el punto de inflexión. Es heurístico y no siempre produce un codo claro.

**Coeficiente de Silueta** (Rousseeuw, 1987): para cada punto, mide qué tan bien encaja en su cluster vs. el cluster más cercano:

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

donde `a(i)` es la distancia media dentro del cluster y `b(i)` la distancia media al cluster vecino más cercano. El coeficiente varía entre −1 (mal agrupado) y +1 (bien agrupado).

---

## 3.2 Reducción de Dimensionalidad

### Contexto teórico

#### La maldición de la dimensionalidad

A medida que la dimensión del espacio crece, los datos se vuelven cada vez más escasos. Bellman (1961) acuñó el término para describir cómo el volumen de un hipercubo crece exponencialmente con la dimensión: para cubrir el 10% del volumen en 10 dimensiones con un hipercubo, se necesita un lado de longitud `0.1^(1/10) ≈ 0.8`, es decir, casi el 80% del rango de cada dimensión.

Consecuencias prácticas: los modelos basados en distancias (kNN, K-Means, SVM-RBF) degradan su rendimiento; la estimación de densidades se vuelve poco confiable; el sobreajuste se intensifica.

#### PCA: Análisis de Componentes Principales

PCA (Pearson, 1901; Hotelling, 1933) encuentra las direcciones de máxima varianza en los datos mediante la descomposición en valores propios de la matriz de covarianza:

```
Σ = V · Λ · Vᵀ
```

donde las columnas de `V` son los eigenvectores (componentes principales) y `Λ` es la matriz diagonal de eigenvalores (varianzas en cada dirección).

La proyección sobre los primeros `k` componentes principales captura la máxima varianza posible en `k` dimensiones, minimizando simultáneamente el error de reconstrucción cuadrático.

#### t-SNE y UMAP: visualización no lineal

**t-SNE** (van der Maaten & Hinton, 2008) minimiza la divergencia KL entre distribuciones de similitud en el espacio original y en el espacio reducido. Es excelente para visualización pero no preserva distancias globales y no es determinista.

**UMAP** (McInnes et al., 2018) es más rápido que t-SNE, preserva mejor la estructura global y produce embeddings deterministas. Se ha convertido en el estándar para visualización de datos de alta dimensión.

---

# Unidad 4
# Optimización de Hiperparámetros e Introducción a Redes Neuronales

---

## 4.1 Optimización de Hiperparámetros

### Contexto teórico

Los hiperparámetros son parámetros del modelo que no se aprenden durante el entrenamiento sino que se fijan antes de él: la profundidad máxima de un árbol, el número de vecinos en kNN, el C de un SVM. Su selección tiene impacto directo en el rendimiento y la capacidad de generalización del modelo.

#### GridSearchCV vs RandomizedSearchCV

**GridSearchCV** evalúa exhaustivamente todas las combinaciones posibles del espacio de hiperparámetros. Si hay `n` hiperparámetros con `m` valores cada uno, el número de evaluaciones es `mⁿ`. Para espacios grandes esto se vuelve computacionalmente inviable.

**RandomizedSearchCV** (Bergstra & Bengio, 2012, *Journal of Machine Learning Research*) muestrea aleatoriamente configuraciones del espacio de búsqueda. Su resultado teórico central es que, si solo el 5% de las configuraciones son "buenas", la probabilidad de encontrar al menos una en 60 pruebas aleatorias es superior al 95%.

#### Validación cruzada estratificada (Stratified K-Fold)

La validación cruzada k-fold divide el dataset en `k` folds, entrena en `k−1` folds y evalúa en el restante, repitiendo el proceso `k` veces. La **versión estratificada** garantiza que la proporción de clases se mantenga igual en cada fold, lo que es esencial en datasets desbalanceados.

```
Score_cv = (1/k) · Σᵢ Score(fold_i)
```

La validación cruzada debe aplicarse **siempre dentro del pipeline**, nunca sobre datos ya transformados, para evitar leakage.

---

## 4.2 Introducción a Redes Neuronales

### Contexto teórico

#### El Perceptrón y sus límites

El Perceptrón (Rosenblatt, 1958) es la neurona artificial más simple: calcula una suma ponderada de sus entradas y aplica una función umbral:

```
y = f(Σᵢ wᵢ · xᵢ + b)
```

Minsky y Papert (1969), en su libro *Perceptrons*, demostraron que el perceptrón simple no puede representar la función XOR, contribuyendo a un período de descrédito de las redes neuronales conocido como el "Invierno de la IA".

#### El Perceptrón Multicapa (MLP) como generalización de la regresión logística

La regresión logística puede interpretarse como un MLP con una sola capa sin capas ocultas y función de activación sigmoide. El MLP generaliza este concepto añadiendo capas ocultas con funciones de activación no lineales:

```
a⁽¹⁾ = f(W⁽¹⁾ · x + b⁽¹⁾)
a⁽²⁾ = f(W⁽²⁾ · a⁽¹⁾ + b⁽²⁾)
ŷ    = softmax(W⁽³⁾ · a⁽²⁾ + b⁽³⁾)
```

El **Teorema de Aproximación Universal** (Cybenko, 1989; Hornik, 1991) garantiza que un MLP con una capa oculta suficientemente ancha puede aproximar cualquier función continua en un compacto con precisión arbitraria.

#### Funciones de activación y su impacto en el gradiente

| Función | Fórmula | Rango | Problema |
|---|---|---|---|
| **Sigmoid** | `σ(x) = 1/(1+e⁻ˣ)` | (0, 1) | Gradientes desvanecientes para \|x\| grande |
| **Tanh** | `tanh(x)` | (−1, 1) | También sufre gradientes desvanecientes |
| **ReLU** | `max(0, x)` | \[0, ∞) | Neuronas muertas (x < 0 siempre inactivo) |
| **Softmax** | `eˣⁱ / Σeˣʲ` | (0,1), suma 1 | Solo para capa de salida multiclase |

El problema del **gradiente desvaneciente** (Hochreiter, 1991) ocurre porque la derivada de sigmoid y tanh es casi cero para valores grandes, haciendo que el gradiente se desvanezca al propagarse hacia capas anteriores. ReLU mitiga este problema al tener derivada constante = 1 para x > 0.

#### Backpropagation: la regla de la cadena aplicada

El algoritmo de backpropagation (Rumelhart, Hinton & Williams, 1986) calcula eficientemente el gradiente de la función de pérdida respecto a todos los pesos mediante la regla de la cadena del cálculo diferencial:

```
∂L/∂w⁽¹⁾ = (∂L/∂a⁽³⁾) · (∂a⁽³⁾/∂a⁽²⁾) · (∂a⁽²⁾/∂a⁽¹⁾) · (∂a⁽¹⁾/∂w⁽¹⁾)
```

El gradiente fluye desde la capa de salida hacia la de entrada, actualizando los pesos mediante gradiente descendiente:

```
wₙₑw = wₒₗd − η · ∂L/∂w
```

#### Prevención de overfitting: Dropout

Dropout (Srivastava et al., 2014, *Journal of Machine Learning Research*) desactiva aleatoriamente una fracción `p` de neuronas durante cada paso de entrenamiento. Esto actúa como regularización implícita: obliga a la red a aprender representaciones robustas que no dependan de ninguna neurona individual. En inferencia, todos los pesos se escalan por `(1−p)`.

---

# Referencias

Las siguientes fuentes académicas sustentan el contenido teórico de estas notas:

**Escalamiento y preprocesamiento:**
- Masini et al. (2025). *The Impact of Feature Scaling in Machine Learning: Effects on Regression and Classification Tasks*. IEEE Access / arXiv:2506.08274.
- Kaufman, S., Rosset, S., & Perlich, C. (2012). *Leakage in Data Mining: Formulation, Detection, and Avoidance*. ACM Transactions on Knowledge Discovery from Data, 6(4), 1–21.
- Little, R. J. A., & Rubin, D. B. (2002). *Statistical Analysis with Missing Data* (2nd ed.). Wiley.
- Yeo, I. K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. *Biometrika*, 87(4), 954–959.

**Desbalance de clases:**
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.
- Kapoor, S., & Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine Learning-based Science. *Patterns*, 4(9).

**Support Vector Machines:**
- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*, 20(3), 273–297.
- Moguerza, J. M., & Muñoz, A. (2006). Support Vector Machines with Applications. *Statistical Science*, 21(3), 322–336.

**Árboles y ensambles:**
- Breiman, L., Friedman, J., Olshen, R., & Stone, C. (1984). *Classification and Regression Trees*. Wadsworth.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*, 785–794.

**Clustering y dimensionalidad:**
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA 2007*, 1027–1035.
- Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters. *KDD 1996*, 226–231.
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *Journal of Computational and Applied Mathematics*, 20, 53–65.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.

**Redes neuronales:**
- Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323, 533–536.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958.
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13, 281–305.
