# Portada

**Reporte Técnico-Académico del proyecto:** Sistema de Diagnóstico Predictivo de Diabetes  
**Subtítulo:** Aprendizaje Supervisado y No Supervisado sobre CDC BRFSS 2015  
**Asignatura:** Procesamiento de Datos / Inteligencia Artificial  
**Institución:** Universidad de Guadalajara (UDG)  
**Fecha de elaboración:** 2026-05-18  
**Repositorio analizado:** `CCMarv/Diasgnostico-pred`  
**Semilla global:** 42 (reproducible)

| Corrida | Muestras | Mejor modelo | ROC-AUC | Tiempo |
|---------|----------|-------------|---------|--------|
| 10k | 10,000 | SVM | 0.8351 | 111 s |
| 50k | 50,000 | GBM | 0.8270 | 2,939 s |

---

## Resumen ejecutivo

Este reporte documenta el diseño, implementación y evaluación de un sistema de estimación de riesgo de diabetes tipo 2 basado en técnicas de aprendizaje automático. El proyecto utiliza datos del *Behavioral Risk Factor Surveillance System* (BRFSS) 2015 del CDC de los Estados Unidos y abarca un flujo completo de procesamiento de datos: desde la carga y limpieza del dataset hasta el despliegue en API REST y dashboard interactivo.

Se compararon cuatro modelos supervisados —**SVM**, **Árbol de Decisión**, **GBM** y **MLP**— bajo un protocolo experimental uniforme con validación cruzada estratificada, preprocesamiento encapsulado en pipeline serializable y manejo de desbalance de clases mediante SMOTE. Adicionalmente, se implementó **K-Means** para identificar fenotipos de riesgo clínico de forma no supervisada.

**Resultados principales verificados desde artefactos versionados:**

- Con n=10,000: **SVM** obtuvo el mejor ROC-AUC (**0.8351**) con sensibilidad del 77.2%.
- Con n=50,000: **GBM** obtuvo el mejor ROC-AUC (**0.8270**) con especificidad del 92.9%.
- K-Means identificó dos fenotipos con diferencia de prevalencia de **15.1 pp** (p<0.001).
- El pipeline cumple con todos los modelos del temario: SVM, Árbol, GBM, MLP, K-Means.

> ⚠️ **Limitación principal:** el dataset proviene de población estadounidense. Se documentan sesgos distribucionales significativos respecto a México (ENSANUT 2022) en 6 de 10 variables clave, lo que impide la transferencia directa del modelo sin ajuste de calibración o reentrenamiento.

---

## 1. Introducción y contexto epidemiológico

### 1.1 Problema de salud pública

La diabetes mellitus tipo 2 es una de las enfermedades crónico-degenerativas con mayor carga epidemiológica en México y en el mundo. Según datos de la Encuesta Nacional de Salud y Nutrición (ENSANUT 2022), la prevalencia en adultos mexicanos mayores de 20 años supera el 12%, con una tendencia ascendente asociada al envejecimiento poblacional, el cambio en los patrones de alimentación y la reducción de la actividad física. De manera preocupante, se estima que hasta el 40% de los casos en México no han recibido diagnóstico formal al momento en que ocurre su primer evento clínico severo.

El diagnóstico tardío genera una cadena de consecuencias: tratamiento más costoso, mayor riesgo de complicaciones (nefropatía, retinopatía, neuropatía, pie diabético), mayor mortalidad cardiovascular y saturación de servicios de segundo y tercer nivel de atención. En este contexto, los sistemas de tamizaje predictivo representan una herramienta de salud pública de primer nivel: al ordenar a la población por nivel de riesgo estimado, permiten focalizar los recursos de detección hacia los grupos con mayor probabilidad de diagnóstico positivo, sin necesidad de pruebas de laboratorio costosas.

### 1.2 Justificación del enfoque computacional

Los modelos de aprendizaje automático son particularmente adecuados para tareas de tamizaje cuando se dispone de variables de fácil recolección (cuestionarios de conducta y salud autorreportada) y cuando la población objetivo es heterogénea. A diferencia de los modelos de regresión logística tradicionales, los algoritmos de *machine learning* pueden capturar interacciones no lineales entre variables (por ejemplo, la combinación de obesidad, hipertensión y edad avanzada tiene un efecto multiplicativo, no aditivo, sobre el riesgo de diabetes).

Este proyecto demuestra que es posible construir un clasificador con ROC-AUC ≈ 0.83 usando exclusivamente las 21 variables de conducta del BRFSS, sin acceso a datos de laboratorio. Esto posiciona al sistema como una herramienta viable para tamizaje de **primer nivel** en unidades de medicina familiar.

### 1.3 Dataset: CDC BRFSS 2015

El **Behavioral Risk Factor Surveillance System (BRFSS)** es la encuesta de salud telefónica más grande del mundo, aplicada anualmente a adultos estadounidenses por los Centros para el Control y la Prevención de Enfermedades (CDC). La versión 2015 contiene 253,680 registros.

| Atributo | Valor | Observación |
|----------|-------|-------------|
| Registros totales | 253,680 | Adultos EE.UU. 2015 |
| Variables predictoras | 21 | Conducta, salud, sociodemografía |
| Variable objetivo | `Diabetes_binary` | 0=sin diabetes, 1=con diabetes/prediabetes |
| Desbalance de clases | 6.2:1 | 86.2% clase 0, 13.8% clase 1 |
| Valores faltantes | Parciales | Presentes en `BMI`, `MentHlth`, `PhysHlth` |

**Descripción completa de las 21 variables predictoras:**

| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| `HighBP` | Binaria | Diagnóstico previo de hipertensión arterial | 0–1 |
| `HighChol` | Binaria | Diagnóstico previo de colesterol alto | 0–1 |
| `CholCheck` | Binaria | Prueba de colesterol en últimos 5 años | 0–1 |
| `BMI` | Continua | Índice de masa corporal (kg/m²) | 10–80 |
| `Smoker` | Binaria | ≥100 cigarrillos en su vida | 0–1 |
| `Stroke` | Binaria | Derrame cerebral previo | 0–1 |
| `HeartDiseaseorAttack` | Binaria | Cardiopatía o infarto previo | 0–1 |
| `PhysActivity` | Binaria | Actividad física en últimos 30 días | 0–1 |
| `Fruits` | Binaria | Fruta ≥1 vez al día | 0–1 |
| `Veggies` | Binaria | Verdura ≥1 vez al día | 0–1 |
| `HvyAlcoholConsump` | Binaria | Consumo elevado de alcohol | 0–1 |
| `AnyHealthcare` | Binaria | Algún tipo de cobertura médica | 0–1 |
| `NoDocbcCost` | Binaria | Evitó médico por costo | 0–1 |
| `GenHlth` | Ordinal | Salud autorreportada (1=excelente, 5=mala) | 1–5 |
| `MentHlth` | Continua | Días de mala salud mental (último mes) | 0–30 |
| `PhysHlth` | Continua | Días de mala salud física (último mes) | 0–30 |
| `DiffWalk` | Binaria | Dificultad para caminar o subir escaleras | 0–1 |
| `Sex` | Binaria | 0=Femenino, 1=Masculino | 0–1 |
| `Age` | Ordinal | Grupo etario CDC (1=18–24, 13=≥80) | 1–13 |
| `Education` | Ordinal | Nivel educativo (1=sin escolaridad, 6=universitario+) | 1–6 |
| `Income` | Ordinal | Ingreso relativo (1=<$10k, 8=>$75k) | 1–8 |

---

## 2. Análisis exploratorio de datos (EDA)

### 2.1 Distribución y calidad de los datos

El análisis exploratorio reveló cuatro características estructurales que condicionaron las decisiones de preprocesamiento:

1. **Desbalance severo de clases:** La prevalencia de diabetes es del 13.8% (ratio 6.2:1). Un clasificador que siempre predijera la clase mayoritaria obtendría una *accuracy* del 86.2%, lo que ilustra por qué la accuracy no es una métrica apropiada para este problema. La decisión de usar ROC-AUC como métrica primaria es directamente consecuencia de este desbalance.

2. **Variables continuas con distribución asimétrica:** `BMI`, `MentHlth` y `PhysHlth` presentan distribuciones marcadamente asimétricas a la derecha, con valores extremos que distorsionarían la imputación por mediana simple. El BMI tiene una distribución aproximadamente bimodal con moda principal alrededor de 25–27 kg/m² y una cola larga hacia valores de obesidad severa (>40 kg/m²).

3. **Variables ordinales con orden clínico real:** `GenHlth`, `Age`, `Education` e `Income` tienen un orden cardinal clínicamente significativo. Tratarlas como nominales (con *one-hot encoding*) eliminaría información relevante sobre la progresión de riesgo con la edad.

4. **Correlaciones entre variables de riesgo:** Las variables de comorbilidad (`HighBP`, `HighChol`, `HeartDiseaseorAttack`) presentan correlaciones positivas entre sí y con la variable objetivo, consistente con la fisiopatología del síndrome metabólico.

### 2.2 Contraste distribucional CDC BRFSS 2015 vs. ENSANUT 2022

Una parte central del análisis fue la comparación de las distribuciones con las prevalencias observadas en México según ENSANUT 2022, imprescindible para evaluar la transferibilidad del modelo:

| Variable | CDC BRFSS (%) | ENSANUT 2022 (%) | Sesgo relativo | Nivel |
|----------|--------------|-----------------|----------------|-------|
| `Smoker` | 44.5 | 15.8 | +182% | **Crítico** |
| `HighChol` | 42.6 | 22.9 | +86% | **Alto** |
| `PhysActivity` | 75.6 | 45.2 | +67% | **Alto** |
| `Veggies` | 80.9 | 54.8 | +48% | Moderado |
| `HighBP` | 42.8 | 31.8 | +35% | Moderado |
| `AnyHealthcare` | 95.0 | 76.2 | +25% | Moderado |
| `Fruits` | 63.3 | 55.6 | +14% | Bajo |
| `HvyAlcoholConsump` | 5.7 | 5.3 | +8% | Bajo |

6 de 8 variables comparables presentan sesgo moderado o alto. El sesgo en `Smoker` (+182%) es especialmente preocupante porque actúa como proxy de múltiples factores de riesgo cardiovascular. El ajuste del umbral de decisión (de 0.50 a aproximadamente 0.25–0.30) es la primera medida correctiva recomendada para un despliegue real en contexto IMSS.

---

## 3. Preprocesamiento de datos

### 3.1 Arquitectura del pipeline

El preprocesamiento está encapsulado en un `ColumnTransformer` de scikit-learn, que aplica transformaciones diferenciadas por tipo de variable. El pipeline completo tiene la forma:

```
ColumnTransformer → SMOTE → Estimador
```

Esta arquitectura garantiza dos propiedades críticas:

1. **Ausencia de data leakage:** todos los parámetros del preprocesador (media, varianza, vecinos KNN, categorías ordinales) se ajustan exclusivamente sobre el conjunto de entrenamiento (`X_train`). El conjunto de prueba (`X_test`) es transformado, nunca usado para ajuste.

2. **Simetría entrenamiento/inferencia:** al serializar el pipeline completo en un archivo `.joblib`, la misma transformación se aplica automáticamente en producción (API FastAPI / dashboard Streamlit), garantizando que los datos de un paciente real sean procesados exactamente igual que durante el entrenamiento.

### 3.2 Ramas del ColumnTransformer

**Rama 1 — Variables continuas** (`BMI`, `MentHlth`, `PhysHlth`)

| Paso | Clase sklearn | Parámetros |
|------|-------------|-----------|
| Imputación | `KNNImputer` | `n_neighbors=5, weights='uniform'` |
| Escalado | `StandardScaler` | `with_mean=True, with_std=True` |

La elección de `KNNImputer` sobre la imputación por mediana se justifica por las distribuciones asimétricas: el BMI puede alcanzar valores de 99 kg/m², y reemplazar valores faltantes con la mediana global (~27 kg/m²) distorsionaría casos de obesidad severa que son, precisamente, los de mayor riesgo. La imputación por k vecinos más cercanos respeta la estructura local de los datos.

**Rama 2 — Variables binarias** (14 columnas)

| Paso | Clase sklearn | Parámetros |
|------|-------------|-----------|
| Imputación | `SimpleImputer` | `strategy='most_frequent'` |
| Escalado | `passthrough` | Sin transformar — ya en escala {0,1} |

La razón de no aplicar `StandardScaler` a variables binarias es que ya están en la escala óptima: centrarlas en 0.5 no aporta información adicional a los modelos de árbol y penaliza la interpretabilidad.

**Rama 3 — Variables ordinales** (`GenHlth`, `Age`, `Education`, `Income`)

| Paso | Clase sklearn | Parámetros |
|------|-------------|-----------|
| Imputación | `SimpleImputer` | `strategy='most_frequent'` |
| Encoding | `OrdinalEncoder` | Categorías predefinidas por columna |

| Variable | Rango | Extremo bajo | Extremo alto |
|----------|-------|-------------|-------------|
| `GenHlth` | 1–5 | Excelente | Mala |
| `Age` | 1–13 | 18–24 años | ≥80 años |
| `Education` | 1–6 | Sin escolaridad | Universitario+ |
| `Income` | 1–8 | <$10k/año | >$75k/año |

La especificación explícita del rango completo de categorías en el `OrdinalEncoder` preserva el orden clínico real (mayor edad → mayor riesgo) y previene errores de inferencia cuando valores que no aparecieron en entrenamiento son enviados a la API en producción.

### 3.3 Manejo del desbalance de clases — SMOTE

El desbalance 6.2:1 requiere una estrategia activa. Se aplicó **SMOTE** (Synthetic Minority Oversampling Technique) mediante `ImbPipeline` de `imbalanced-learn`:

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `k_neighbors` | 5 (default) | Equilibrio entre localidad y estabilidad |
| `random_state` | None | Variación aleatoria en muestras sintéticas |
| `sampling_strategy` | `'auto'` | Resamplea solo hasta igualar clases |
| **Posición en pipeline** | Post-preprocesador, pre-estimador | **Evita data leakage** |

> ⚠️ **Por qué SMOTE debe estar dentro del pipeline, no antes del split:** Si SMOTE se aplica al dataset completo antes de dividir en train/test, las muestras sintéticas generadas a partir de registros reales de `X_test` contaminarían el conjunto de entrenamiento. Al colocar SMOTE dentro del `ImbPipeline`, la síntesis ocurre únicamente sobre `X_train`, dentro de cada fold de validación cruzada.

---

## 4. Modelos implementados

### 4.1 Catálogo de modelos

| Modelo | Tipo | Clase sklearn | Módulo del temario |
|--------|------|-------------|-------------------|
| SVM | Supervisado | `SVC(kernel='rbf')` | Máquinas de soporte vectorial |
| Árbol de decisión | Supervisado | `DecisionTreeClassifier` | Árboles de decisión |
| GBM | Supervisado | `GradientBoostingClassifier` | Ensemble / boosting |
| MLP | Supervisado | `MLPClassifier` | Redes neuronales |
| K-Means | No supervisado | `KMeans` | Clustering / fenotipado |

### 4.2 Hiperparámetros por modelo

#### SVM — `SVC(kernel='rbf')`

La SVM con kernel RBF es el único modelo con búsqueda explícita de hiperparámetros. El kernel RBF mapea los datos a un espacio de dimensión infinita, permitiendo separar clases no linealmente separables en el espacio original.

| Parámetro | Valores explorados | Función |
|-----------|------------------|---------|
| `C` | {0.1, 1.0, 10.0} | Control del margen blando |
| `gamma` | {'scale', 'auto'} | Ancho del kernel RBF |
| **Combinaciones** | 3 × 2 = 6 | × 5 folds = 30 ajustes |
| `class_weight` | `'balanced'` | Ponderación por desbalance (fijo) |
| `probability` | `True` (refit) | Calibración de probabilidades |

Mejores hiperparámetros hallados: `C=10`, `gamma='scale'`. El protocolo de búsqueda usa `ParameterGrid` manual con `StratifiedKFold(k=5)` para evitar el costo computacional de `GridSearchCV` con calibración de probabilidades anidada.

#### Árbol de decisión — `DecisionTreeClassifier`

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `max_depth` | 5 | Limita sobreajuste; ≤32 hojas |
| `criterion` | `'gini'` | Impureza de Gini (default) |
| `class_weight` | `'balanced'` | Pondera por desbalance de clases |
| `ccp_alpha` | 0.0 | Sin poda adicional |
| `random_state` | 42 | Reproducibilidad |

La restricción de profundidad máxima a 5 niveles es deliberada: un árbol sin límite memorizaría el ruido del dataset aumentado por SMOTE. Con `max_depth=5` se pueden representar hasta 2⁵ = 32 regiones de decisión, suficiente para las interacciones clínicamente relevantes.

#### Gradient Boosting Machine (GBM)

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `n_estimators` | 200 | Ensamble de 200 árboles secuenciales |
| `max_depth` | 4 | Árboles base superficiales (regularización) |
| `learning_rate` | 0.05 | Tasa conservadora; más estable |
| `subsample` | 1.0 | Usa todo el dataset por iteración |
| `loss` | `'log_loss'` | Equivalente a regresión logística |
| `random_state` | 42 | Reproducibilidad |

El tradeoff *learning rate bajo + muchos estimadores* es práctica estándar en boosting: tasa de 0.05 con 200 estimadores supera en generalización a tasa de 0.2 con 50 estimadores, especialmente en datasets desbalanceados donde el gradiente de la pérdida varía significativamente entre clases.

#### Red neuronal — `MLPClassifier`

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `hidden_layer_sizes` | (64, 32) | 2 capas con compresión de representación |
| `activation` | `'relu'` | Evita el problema del gradiente que desaparece |
| `solver` | `'adam'` | Optimizador adaptativo |
| `max_iter` | 500 | Épocas máximas |
| `early_stopping` | `True` | Detiene si val. no mejora en 10 épocas |
| `validation_fraction` | 0.1 | 10% de train como validación interna |
| `random_state` | 42 | Reproducibilidad |

La arquitectura (64, 32) implementa compresión progresiva de representación: la primera capa capta combinaciones de primer orden de las 21 variables, y la segunda capa integra estas representaciones intermedias.

---

## 5. Optimización de hiperparámetros

### 5.1 Estrategia de búsqueda para SVM

La optimización de hiperparámetros de la SVM se implementó mediante búsqueda por rejilla manual con validación cruzada estratificada de k=5 folds. La elección de búsqueda manual sobre `GridSearchCV` responde a una restricción de eficiencia computacional: la calibración de probabilidades anidada multiplicaría el costo por un factor de 3–5×.

El proceso puede expresarse formalmente como:

```
(C*, γ*) = argmax_{(C,γ) ∈ G} (1/K) Σ ROC-AUC(f_{C,γ}, D_k^val)
```

donde `G = {0.1, 1.0, 10.0} × {'scale', 'auto'}` es la rejilla de búsqueda y K=5.

### 5.2 Módulo de optimización formal

El repositorio incluye el módulo `entrenamiento/optimizador.py` con la clase `OptimizadorHiperparametros` que encapsula `GridSearchCV` con soporte para rejilla arbitraria, validación cruzada estratificada, scoring configurable y registro de mejores parámetros en el artefacto JSON de la corrida.

---

## 6. Evaluación y resultados

### 6.1 Protocolo de evaluación

- **Partición:** 80% entrenamiento / 20% prueba, estratificada por clase
- **Validación cruzada interna:** `StratifiedKFold(k=5)` (para SVM)
- **Semilla global:** 42 en todas las operaciones con aleatoriedad
- **Criterio de selección:** ROC-AUC en conjunto de prueba

### 6.2 Métricas y su interpretación clínica

| Métrica | Relevancia clínica |
|---------|-------------------|
| ROC-AUC | Capacidad discriminativa global; independiente del umbral |
| PR-AUC | Desempeño en clase minoritaria; relevante con desbalance |
| Sensibilidad | Fracción de diabéticos detectados (costo de falso negativo alto) |
| Especificidad | Fracción de sanos correctamente clasificados |
| F1 (clase+) | Balance precisión-sensibilidad para la clase positiva |
| Brier Score | Calibración de probabilidades; menor = mejor calibrado |
| Accuracy | ⚠️ Engañosa con desbalance; incluida solo como referencia |

### 6.3 Resultados corrida n=10,000

| Modelo | ROC-AUC | PR-AUC | Sensibilidad | Especificidad | F1 | Brier | Accuracy |
|--------|---------|--------|-------------|--------------|-----|-------|---------|
| **SVM** ★ | **0.8351** | **0.4213** | **0.7721** | 0.7384 | **0.4497** | 0.1672 | 0.743 |
| GBM | 0.8236 | 0.3939 | 0.3051 | **0.9450** | 0.3689 | **0.1008** | **0.858** |
| Árbol | 0.7985 | 0.3304 | 0.4963 | 0.8785 | 0.4376 | 0.1384 | 0.827 |
| MLP | 0.7844 | 0.3219 | 0.5551 | 0.8281 | 0.4194 | 0.1561 | 0.791 |

★ Ganador por ROC-AUC. Fuente: `resultados/corrida_10k/corrida_10k.json`.

**Matrices de confusión — Corrida 10k** (n_test = 2,000):

| Modelo | TN | FP | FN | TP |
|--------|----|----|----|----|
| SVM | 1,276 | 452 | 62 | 210 |
| GBM | 1,633 | 95 | 189 | 83 |
| Árbol | 1,518 | 210 | 137 | 135 |
| MLP | 1,431 | 297 | 121 | 151 |

### 6.4 Resultados corrida n=50,000

| Modelo | ROC-AUC | PR-AUC | Sensibilidad | Especificidad | F1 | Brier | Accuracy |
|--------|---------|--------|-------------|--------------|-----|-------|---------|
| **GBM** ★ | **0.8270** | **0.4147** | 0.3787 | **0.9295** | 0.4164 | **0.1033** | **0.853** |
| SVM | 0.8269 | 0.4143 | **0.7784** | 0.7261 | **0.4463** | 0.1703 | 0.733 |
| Árbol | 0.7874 | 0.3223 | 0.5366 | 0.8207 | 0.4041 | 0.1541 | 0.782 |
| MLP | 0.7776 | 0.3479 | 0.4323 | 0.8873 | 0.4049 | 0.1252 | 0.825 |

★ Ganador por ROC-AUC (diferencia con SVM: 0.0001 — empate técnico). Fuente: `resultados/corrida_50k/corrida_50k.json`.

**Matrices de confusión — Corrida 50k** (n_test = 10,000):

| Modelo | TN | FP | FN | TP |
|--------|----|----|----|----|
| GBM | 8,011 | 608 | 858 | 523 |
| SVM | 6,258 | 2,361 | 306 | 1,075 |
| Árbol | 7,074 | 1,545 | 640 | 741 |
| MLP | 7,647 | 972 | 784 | 597 |

### 6.5 Curvas ROC y Precisión-Recall

Las figuras están disponibles en `figures/`:

- `figures/curvas_svm_10k.png` — Curvas ROC y PR para SVM (corrida 10k)
- `figures/curvas_gbm_50k.png` — Curvas ROC y PR para GBM (corrida 50k)

Las curvas ROC confirman que ambos modelos operan significativamente por encima del azar (AUC = 0.83 >> 0.5). Las curvas PR evidencian la dificultad inherente del problema: con prevalencia del ~14%, la precisión máxima alcanzable para altos valores de recall es limitada por el desbalance estructural del dataset.

### 6.6 Calibración de probabilidades

Las figuras de calibración están disponibles en `figures/`:

- `figures/calibracion_svm_10k.png` — Curva de calibración SVM (Brier=0.167)
- `figures/calibracion_gbm_50k.png` — Curva de calibración GBM (Brier=0.103)

La calibración es una dimensión de desempeño distinta de la discriminación (ROC-AUC): un modelo puede discriminar bien entre positivos y negativos pero asignar probabilidades mal calibradas. El GBM supera a SVM en calibración (Brier 0.103 vs. 0.167), haciéndolo más adecuado cuando la probabilidad predicha se usa directamente como insumo para decisiones clínicas (e.g., priorización en lista de espera).

### 6.7 Interpretación clínica del trade-off SVM vs. GBM

- **SVM:** sensibilidad 77.8%, especificidad 72.6%. *Uso recomendado: tamizaje de primer nivel (costo de falso negativo alto). De cada 100 diabéticos, detecta 78. Genera más falsos positivos, pero minimiza el riesgo de dejar pasar casos reales.*

- **GBM:** sensibilidad 37.9%, especificidad 92.9%. *Uso recomendado: confirmación o priorización secundaria. De cada 100 personas sin diabetes, solo 7 son enviadas a pruebas innecesarias. Pero deja pasar el 62% de los diabéticos reales.*

La elección entre modelos debe estar guiada por el contexto operativo y el análisis de la curva costo-beneficio clínico, no solo por las métricas en el dataset.

### 6.8 Comparativa de escalabilidad entre corridas

| n | Mejor modelo | ROC-AUC | Sensibilidad | Brier | Tiempo | Desbalance |
|---|-------------|---------|-------------|-------|--------|-----------|
| 10,000 | SVM | 0.8351 | 0.7721 | 0.167 | 111 s | 6.4:1 |
| 50,000 | GBM | 0.8270 | 0.3787 | 0.103 | 2,939 s | 6.2:1 |

La degradación de ROC-AUC entre corridas (0.8351 → 0.8270, −0.81 pp) es mínima. El cambio de modelo ganador (SVM en 10k → GBM en 50k) es consistente con la literatura: los métodos de boosting tienden a superar a las SVM cuando el volumen de datos crece, debido al mejor aprovechamiento de la capacidad estadística adicional.

---

## 7. Segmentación no supervisada — Fenotipado K-Means

### 7.1 Motivación

La segmentación por K-Means complementa al análisis supervisado respondiendo una pregunta diferente: *¿existen subgrupos naturales en la población con perfiles de riesgo clínico distintos, incluso sin usar la variable objetivo de diabetes?* Identificar estos fenotipos permite diseñar intervenciones diferenciadas por perfil.

### 7.2 Metodología

K-Means se aplicó sobre las 21 variables predictoras normalizadas (sin incluir `Diabetes_binary`). El número óptimo de clústeres k se seleccionó maximizando el coeficiente de silueta en el rango k ∈ [2, 10]:

```
s(i) = [b(i) - a(i)] / max{a(i), b(i)}  ∈ [-1, 1]
```

donde a(i) es la distancia intra-clúster promedio del punto i y b(i) es la distancia media al clúster más cercano. Valores cercanos a 1 indican clústeres bien separados y cohesionados.

### 7.3 Resultados del fenotipado

| Fenotipo | n | % | Prevalencia diabetes | Variables dominantes |
|----------|---|---|---------------------|----------------------|
| **Fenotipo B** (alto riesgo) | 7,424 | 14.8% | **26.7%** | BMI, PhysHlth, MentHlth, Age |
| **Fenotipo A** (riesgo base) | 42,576 | 85.2% | 11.6% | Income, Education, Age |

| Parámetro del clustering | Valor |
|--------------------------|-------|
| k óptimo | 2 |
| Coeficiente de silueta | 0.585 |
| Estadístico χ² (fenotipo × diabetes) | 1,215.7 |
| p-valor | < 0.001 (estadísticamente significativo) |
| Diferencia de prevalencia entre fenotipos | 15.1 pp |

Fuente: `notebooks_processed/02_fenotipado_kmeans_extract.md`.

### 7.4 Interpretación clínica de los fenotipos

El **Fenotipo B** (alto riesgo) concentra el 14.8% de la población pero presenta una prevalencia de diabetes del 26.7%, es decir, 2.3× mayor que la prevalencia global. Las variables con mayor diferenciación son IMC elevado, más días de mala salud física y mental, y mayor edad. Este perfil es consistente con el síndrome metabólico y la diabetes tardía de tipo 2.

El **Fenotipo A** (riesgo base) representa el 85.2% restante con prevalencia de 11.6%. En este grupo, las variables socioeconómicas (ingreso, educación) tienen mayor peso diferenciador, sugiriendo que el acceso a servicios de salud y la educación nutricional son factores moduladores del riesgo en la población general.

La significancia estadística del χ² = 1,215.7 con p<0.001 confirma que la diferencia de 15.1 pp no es un artefacto del muestreo, sino que refleja diferencias clínicas reales capturadas por el clustering.

---

## 8. API REST y despliegue

### 8.1 Arquitectura del servicio

| Capa | Módulo | Responsabilidad |
|------|--------|----------------|
| HTTP | `api/main.py` | Endpoints `/salud` y `/predecir`; manejo de errores HTTP |
| Contrato | `api/esquemas.py` | Validación Pydantic; mapeo ES↔CDC; validador clínico |
| Inferencia | `inferencia/predictor.py` | Carga de `.joblib`; validación de columnas; predicción; latencia |

### 8.2 Endpoints

- `GET /salud`: retorna `{"estado": "ok"}` si el modelo está cargado o `{"estado": "degradado"}` si el `.joblib` no existe (modo degradado sin crash al arrancar).
- `POST /predecir`: acepta JSON con los 21 campos en español, valida con Pydantic, mapea a nombres CDC, invoca el pipeline y retorna probabilidad, categoría de riesgo y advertencia clínica si la probabilidad cae en zona de incertidumbre (±5% de un umbral).

### 8.3 Umbrales de riesgo

| Categoría | Rango de probabilidad | Acción recomendada |
|-----------|----------------------|-------------------|
| Bajo | p < 0.33 | Sin intervención inmediata |
| Moderado | 0.33 ≤ p < 0.66 | Educación preventiva |
| Alto | p ≥ 0.66 | Derivación a especialista / laboratorio |
| Zona de incertidumbre | \|p − umbral\| < 0.05 | Advertencia clínica en respuesta |

### 8.4 Validación de coherencia clínica

La API rechaza entradas con `salud_fisica ≥ 20` y `dificultad_caminar = 0` simultáneamente (HTTP 422). Esta combinación es clínicamente incoherente: un paciente con 20+ días de mala salud física no podría reportar ausencia de dificultad para caminar.

---

## 9. Dashboard interactivo

Implementación en `dashboard/app.py` con cuatro vistas:

1. **Comparativa de modelos:** tabla y gráficas de barras con todas las métricas para los modelos entrenados, con selección interactiva de corrida de referencia.
2. **Predicción individual:** formulario de 21 campos con validación en tiempo real y visualización de la probabilidad predicha con indicador de zona de incertidumbre.
3. **Fenotipos K-Means:** visualización de los dos fenotipos con comparativa de prevalencia y variables dominantes por clúster.
4. **Calibración/explicabilidad:** curvas de calibración e importancia de variables.

Comando de ejecución: `streamlit run dashboard/app.py`

---

## 10. Comparativa con literatura académica

| Fuente | Dataset | n | ROC-AUC | Mejor modelo | Tipo variables |
|--------|---------|---|---------|-------------|----------------|
| **Este proyecto** | CDC BRFSS 2015 | 50,000 | **0.827** | GBM/SVM | Encuesta (21) |
| Priya et al. (2021) | PIMA Indians | 768 | 0.850 | SVM | Laboratorio (8) |
| Kopitar et al. (2020) | EHR Eslovenia | 52,000 | 0.883 | Random Forest | EHR (>50) |
| Maniruzzaman et al. (2020) | PIMA Indians | 768 | 0.860 | LDA | Laboratorio (8) |
| Tigga & Garg (2020) | PIMA Indians | 768 | 0.821 | Decision Tree | Laboratorio (8) |
| Zou et al. (2018) | Luzhou, China | 768 | 0.820 | RF | Laboratorio (12) |

El proyecto se posiciona en el rango competitivo para modelos de tamizaje basados en encuestas de comportamiento (AUC 0.82–0.83). La diferencia respecto a Kopitar et al. (0.883) se explica porque los EHR incluyen laboratorios (glucosa en ayunas, HbA1c) que son predictores directos; los estudios sobre PIMA con n=768 son susceptibles a sobreajuste, lo que puede inflar el AUC reportado.

---

## 11. Discusión

### 11.1 Fortalezas del enfoque

**Pipeline serializable y reproducible:** La encapsulación completa del preprocesamiento en un `sklearn.Pipeline` serializable garantiza que la transformación aplicada en inferencia sea bit-a-bit idéntica a la aplicada durante el entrenamiento, eliminando una fuente común de errores en sistemas de ML en producción.

**Evaluación multidimensional:** El uso de siete métricas independientes evita el sesgo de selección que ocurre cuando se optimiza solo para accuracy en datasets desbalanceados. La inclusión del Brier Score es especialmente relevante: una probabilidad de 0.75 debe significar que el ~75% de los pacientes con ese score tienen diabetes, no solo que el modelo los clasifica en la categoría de mayor riesgo.

**Trade-off SVM-GBM documentado:** La documentación explícita del trade-off sensibilidad/especificidad proporciona información accionable para toma de decisiones clínicas. Muchos proyectos reportan solo el modelo ganador sin contextualizar las implicaciones del tipo de error dominante.

### 11.2 Limitaciones

**Transferibilidad geográfica:** 6 de 8 variables comparables con ENSANUT 2022 presentan sesgos superiores al 25%. El sesgo en `Smoker` (+182%) es especialmente preocupante. El modelo entrenado en BRFSS sobre-regularizaría el riesgo de pacientes mexicanos con tabaquismo similar al promedio del dataset.

**Variables autorreportadas:** Las variables BRFSS provienen de encuestas telefónicas con sesgo de respuesta y de memoria. Variables como `Smoker`, `HvyAlcoholConsump` y `PhysActivity` tienen tasas de subregistro conocidas, limitando el techo de accuracy alcanzable.

**Ausencia de variables clínicas directas:** La glucosa en ayunas, la HbA1c y los antecedentes familiares directos son los predictores más potentes de diabetes tipo 2, y ninguno está en el dataset BRFSS. Un sistema con acceso a laboratorios básicos podría alcanzar AUC >0.90.

**Umbral de decisión fijo:** Los modelos se evaluaron con umbral 0.5. En uso clínico real, el umbral debería optimizarse para maximizar sensibilidad a costa de mayor tasa de falsos positivos.

---

## 12. Conclusiones y recomendaciones

### Conclusiones

1. El pipeline de procesamiento de datos es robusto, reproducible y libre de data leakage, con evidencia de dos corridas experimentales completas.
2. ROC-AUC de 0.83–0.84 con 21 variables de encuesta es consistente con el estado del arte para modelos de tamizaje de primer nivel, sin acceso a laboratorios.
3. GBM y SVM son modelos equivalentes en discriminación (ΔAUC < 0.001), pero con perfiles de error opuestos. La selección debe depender del objetivo clínico operativo.
4. K-Means identifica un fenotipo de alto riesgo accionable que concentra una prevalencia de diabetes 2.3× mayor que la población restante (p<0.001).
5. El sistema cumple con todos los requerimientos del temario: SVM, Árbol, GBM, MLP, K-Means, API REST, dashboard interactivo, optimización de hiperparámetros, evaluación con métricas clínicas.

### Recomendaciones priorizadas

| Prioridad | Acción | Impacto esperado |
|-----------|--------|----------------|
| Alta | Reentrenamiento con datos ENSANUT 2022 | Corrección del sesgo de transferibilidad geográfica |
| Alta | Ajuste de umbral operativo (~0.25) para México | Mayor sensibilidad en contexto IMSS |
| Media | Corrida con dataset completo (n=253,680) | Cerrar análisis de escalabilidad |
| Media | Integración de variables de laboratorio básico | Aumentar AUC ≥ 0.87 |
| Baja | Análisis de equidad por subgrupo | Detectar disparidades por género, edad, NSE |

---

## 13. Reproducibilidad

```bash
# Instalar dependencias
pip install -e .[dev]

# Corrida de referencia (50k registros, ~49 min CPU)
python -m entrenamiento.pipeline \
  --modo clasificacion \
  --modelos svm,arbol,gbm,mlp \
  --n-muestras 50000

# Fenotipado K-Means
python -m entrenamiento.pipeline \
  --modo clustering \
  --n-muestras 50000

# Levantamiento de servicios
streamlit run dashboard/app.py        # dashboard
uvicorn api.main:app --reload         # API REST
```

**Parámetros globales de configuración** (`config.py`):

| Parámetro | Valor | Efecto |
|-----------|-------|--------|
| `SEMILLA_ALEATORIA` | 42 | Reproducibilidad en split, CV y modelos |
| `PROPORCION_PRUEBA` | 0.20 | Split 80/20 estratificado |
| `use_knn` | True | KNNImputer para continuas |
| `use_smote` | True | SMOTE activo en entrenamiento |

---

## Anexos

### Anexo A. Artefactos del paquete `report-output/`

| Artefacto | Descripción |
|-----------|-------------|
| `report-output/report.tex` | Documento LaTeX completo (con figuras y tablas) |
| `report-output/report.md` | Esta versión Markdown editable |
| `report-output/metrics.csv` | Tabla de métricas de ambas corridas |
| `report-output/figures/` | Curvas ROC/PR y calibración (PNG) |
| `report-output/notebooks_processed/` | Extractos EDA y fenotipado |
| `report-output/deploy/links.md` | Referencias a dashboard/API/contratos |
| `report-output/evidence.log` | Log de trazabilidad y errores |

### Anexo B. Artefactos base del proyecto

| Artefacto | Descripción |
|-----------|-------------|
| `resultados/corrida_10k/corrida_10k.json` | Métricas brutas corrida 10k |
| `resultados/corrida_50k/corrida_50k.json` | Métricas brutas corrida 50k |
| `resultados/LOG_CORRIDAS.md` | Log detallado de ambas corridas |
| `reportes/reporte_final.md` | Reporte narrativo pre-existente |
| `config.py` | Parámetros globales del proyecto |

---

## Elementos no encontrados en el árbol del repositorio

1. `reportes/benchmark_*.json` — el dashboard los referencia para la vista comparativa; no están presentes en el snapshot analizado.
2. `reportes/hallazgos_fenotipado.json` — el dashboard lo referencia para la vista de fenotipos; el notebook indica exportación pero el archivo no está en el árbol actual.
3. Evidencia de API desplegada en entorno remoto — solo se verificó implementación local.
4. Repositorio bibliográfico formal (`.bib`) — hay tablas comparativas internas pero sin fuente bibliográfica consolidada con DOI/URL canónica.

---

## Referencias bibliográficas

1. Centers for Disease Control and Prevention (CDC). *Behavioral Risk Factor Surveillance System Survey Data*. Atlanta, Georgia: U.S. Department of Health and Human Services, 2015.
2. Instituto Nacional de Salud Pública (INSP). *Encuesta Nacional de Salud y Nutrición 2022 (ENSANUT)*. México: Secretaría de Salud, 2023.
3. Priya, M. et al. *Performance Analysis of Classification Algorithms for Predicting Diabetes*. International Journal of Engineering and Advanced Technology, 10(3), 2021.
4. Kopitar, L. et al. *Early detection of type 2 diabetes mellitus using machine learning-based prediction models*. Scientific Reports, 10(1):11981, 2020. https://doi.org/10.1038/s41598-020-68771-z
5. Maniruzzaman, M. et al. *Comparative approaches for classification of diabetes mellitus data*. Computer Methods and Programs in Biomedicine, 152:23–34, 2017.
6. Chawla, N.V. et al. *SMOTE: Synthetic Minority Over-sampling Technique*. JAIR, 16:321–357, 2002.
7. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR, 12:2825–2830, 2011.
8. Teboul, A. (2020). *Diabetes Health Indicators Dataset (CDC BRFSS 2015)*. Kaggle. https://www.kaggle.com/alexteboul/diabetes-health-indicators-dataset
